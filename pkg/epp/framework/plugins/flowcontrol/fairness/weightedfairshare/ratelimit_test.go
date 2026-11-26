/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package weightedfairshare

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time          { return c.t }
func (c *fakeClock) advance(d time.Duration) { c.t = c.t.Add(d) }

// newRatedPolicy builds a policy from params with an injected clock.
func newRatedPolicy(t *testing.T, params string) (*weightedFairShare, *fakeClock) {
	t.Helper()
	p := newPolicy(t, params)
	clk := &fakeClock{t: time.Unix(1_000_000, 0)}
	p.now = clk.now
	return p, clk
}

// request is one queued request of flow id at priority.
func request(id string, priority int, reqID string, sizeBytes int) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID:        reqID,
		FairnessID:       id,
		RequestSizeBytes: sizeBytes,
		Objectives:       fwksched.RequestObjectives{Priority: priority},
	}
}

// queueOf is a one-request queue holding req, enqueued at enqueued.
func queueOf(req *fwksched.InferenceRequest, enqueued time.Time) flowcontrol.FlowQueueAccessor {
	key := flowcontrol.FlowKey{ID: req.FairnessID, Priority: req.Objectives.Priority}
	return &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: key,
		PeekV: &fwkfcmocks.MockQueueItemAccessor{
			EnqueueTimeV: enqueued,
			OriginalRequestV: &fwkfcmocks.MockFlowControlRequest{
				FlowKeyV:          key,
				ByteSizeV:         uint64(req.RequestSizeBytes),
				InferenceRequestV: req,
			},
		},
	}
}

// finish reports end of stream for req with the given usage.
func finish(p *weightedFairShare, req *fwksched.InferenceRequest, prompt, cached, completion int) {
	resp := &fwkrh.Response{EndOfStream: true}
	resp.Usage.PromptTokens = prompt
	resp.Usage.CompletionTokens = completion
	if cached > 0 {
		resp.Usage.PromptTokenDetails = &fwkrequest.PromptTokenDetails{CachedTokens: cached}
	}
	p.ResponseBody(context.Background(), req, resp, nil)
}

func balance(t *testing.T, p *weightedFairShare, id string) float64 {
	t.Helper()
	v, ok := p.rates.Load(id)
	require.True(t, ok, "no bucket for %q", id)
	st := v.(*rateState)
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.refill(p.now())
}

func pickID(t *testing.T, p *weightedFairShare, b flowcontrol.PriorityBandAccessor) string {
	t.Helper()
	got, err := p.Pick(context.Background(), b)
	require.NoError(t, err)
	if got == nil {
		return ""
	}
	return got.FlowKey().ID
}

func TestConfig_RejectsInvalidRates(t *testing.T) {
	tests := []struct {
		name   string
		params string
	}{
		{name: "negative_rate", params: `{"flowRates": {"a": {"rate": -1}}}`},
		{name: "negative_burst", params: `{"flowRates": {"a": {"rate": 1, "burst": -1}}}`},
		{name: "burst_without_rate", params: `{"flowRates": {"a": {"burst": 10}}}`},
		{name: "negative_default_rate", params: `{"defaultFlowRate": {"rate": -5}}`},
		{name: "negative_cached_input_cost", params: `{"costPerCachedInputToken": -0.1}`},
		{name: "negative_initial_output_tokens", params: `{"initialOutputTokens": -1}`},
		{name: "zero_bytes_per_input_token", params: `{"bytesPerInputToken": 0}`},
		{name: "zero_pending_ttl", params: `{"pendingTtlSeconds": 0}`},
		{name: "negative_max_hold", params: `{"maxHoldSeconds": -1}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dec := json.NewDecoder(bytes.NewReader([]byte(tc.params)))
			_, err := WeightedFairShareFairnessPolicyFactory("test", dec, nil)
			assert.Error(t, err)
		})
	}
}

func TestConfig_BurstDefaultsToTenSecondsOfRate(t *testing.T) {
	p := newPolicy(t, `{"flowRates": {"a": {"rate": 30}, "b": {"rate": 30, "burst": 5}}, "defaultFlowRate": {"rate": 2}}`)
	assert.Equal(t, FlowRate{Rate: 30, Burst: 300}, p.rateFor("a"))
	assert.Equal(t, FlowRate{Rate: 30, Burst: 5}, p.rateFor("b"))
	assert.Equal(t, FlowRate{Rate: 2, Burst: 20}, p.rateFor("unlisted"))
}

func TestCost_CachedInputTokens(t *testing.T) {
	tests := []struct {
		name   string
		params string
		want   float64
	}{
		// 1000 prompt tokens, 800 of them cached, 10 completion tokens.
		{name: "cached_free", params: `{"costPerInputToken": 1, "costPerCachedInputToken": 0, "costPerOutputToken": 1}`, want: 210},
		{name: "cached_discounted", params: `{"costPerInputToken": 1, "costPerCachedInputToken": 0.1, "costPerOutputToken": 1}`, want: 290},
		{name: "unset_prices_cached_as_input", params: `{"costPerInputToken": 1, "costPerOutputToken": 1}`, want: 1010},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPolicy(t, tc.params)
			finish(p, request("a", 0, "r1", 0), 1000, 800, 10)
			assert.InDelta(t, tc.want, virtualService(p, "a"), 1e-9)
		})
	}
}

func TestCost_CachedTokensClampedToPrompt(t *testing.T) {
	p := newPolicy(t, `{"costPerInputToken": 1, "costPerCachedInputToken": 0, "costPerOutputToken": 1}`)
	finish(p, request("a", 0, "r1", 0), 100, 500, 10)
	assert.InDelta(t, 10.0, virtualService(p, "a"), 1e-9)
}

func TestPick_SkipsFlowOverCeiling(t *testing.T) {
	// Seed estimate for a 4000-byte request: 4000/4 * 1 input + 256 * 2 output = 1512 > burst 100.
	p, clk := newRatedPolicy(t, `{"flowRates": {"capped": {"rate": 10, "burst": 100}}}`)
	state := &minVirtualService{}
	old := clk.t.Add(-time.Second)

	got := pickID(t, p, band(state, queueOf(request("capped", 0, "c1", 4000), old), queueOf(request("free", 0, "f1", 4000), clk.t)))
	require.Equal(t, "capped", got)
	assert.InDelta(t, 100.0-1512.0, balance(t, p, "capped"), 1e-9)

	// capped still has the older head and an equal virtual service, but its balance is negative.
	got = pickID(t, p, band(state, queueOf(request("capped", 0, "c2", 4000), old), queueOf(request("free", 0, "f2", 4000), clk.t)))
	assert.Equal(t, "free", got)
}

func TestPick_CeilingHoldsWhenOnlyCappedFlowHasWork(t *testing.T) {
	p, clk := newRatedPolicy(t, `{"flowRates": {"capped": {"rate": 100, "burst": 100}}, "costPerInputToken": 0, "costPerOutputToken": 1}`)
	state := &minVirtualService{}

	require.Equal(t, "capped", pickID(t, p, band(state, queueOf(request("capped", 0, "c1", 0), clk.t))))
	// Seed output estimate 256 leaves the balance at -156.
	require.InDelta(t, -156.0, balance(t, p, "capped"), 1e-9)

	assert.Empty(t, pickID(t, p, band(state, queueOf(request("capped", 0, "c2", 0), clk.t))))
	clk.advance(1500 * time.Millisecond) // +150 -> -6
	assert.Empty(t, pickID(t, p, band(state, queueOf(request("capped", 0, "c2", 0), clk.t))))
	clk.advance(100 * time.Millisecond) // +10 -> 4
	assert.Equal(t, "capped", pickID(t, p, band(state, queueOf(request("capped", 0, "c2", 0), clk.t))))
}

func TestPick_RepickChargesOnce(t *testing.T) {
	p, clk := newRatedPolicy(t, `{"flowRates": {"capped": {"rate": 1, "burst": 10000}}, "costPerInputToken": 0, "costPerOutputToken": 1}`)
	state := &minVirtualService{}
	req := request("capped", 0, "c1", 0)

	for range 3 {
		require.Equal(t, "capped", pickID(t, p, band(state, queueOf(req, clk.t))))
	}
	assert.InDelta(t, 10000.0-256.0, balance(t, p, "capped"), 1e-9)
}

func TestSettle_ChargesActualCost(t *testing.T) {
	tests := []struct {
		name     string
		dispatch bool
	}{
		{name: "estimate_replaced_by_actual", dispatch: true},
		{name: "no_estimate_charges_full_cost", dispatch: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, clk := newRatedPolicy(t, `{"flowRates": {"capped": {"rate": 1, "burst": 10000}}, "costPerInputToken": 0.5, "costPerOutputToken": 1}`)
			req := request("capped", 0, "c1", 800)
			if tc.dispatch {
				require.Equal(t, "capped", pickID(t, p, band(&minVirtualService{}, queueOf(req, clk.t))))
			} else {
				p.rateStateFor("capped", clk.t)
			}
			finish(p, req, 200, 0, 40)
			assert.InDelta(t, 10000.0-(0.5*200+40), balance(t, p, "capped"), 1e-9)
			_, pending := p.pending.Load("c1")
			assert.False(t, pending)
		})
	}
}

func TestEstimate_LearnsFromCompletions(t *testing.T) {
	p, clk := newRatedPolicy(t, `{"flowRates": {"capped": {"rate": 1, "burst": 100000}}, "costPerInputToken": 1, "costPerCachedInputToken": 0, "costPerOutputToken": 1}`)
	// 4000 bytes, 1000 prompt tokens of which 750 cached, 100 completion: input 250, output 100.
	finish(p, request("capped", 0, "c1", 4000), 1000, 750, 100)
	before := balance(t, p, "capped")

	require.Equal(t, "capped", pickID(t, p, band(&minVirtualService{}, queueOf(request("capped", 0, "c2", 8000), clk.t))))
	// First sample replaces the seeds: 250/4000 per byte * 8000 + 100.
	assert.InDelta(t, 600.0, before-balance(t, p, "capped"), 1e-9)
}

func TestEstimate_CappedByMaxOutputTokens(t *testing.T) {
	p, clk := newRatedPolicy(t, `{"flowRates": {"capped": {"rate": 1, "burst": 1000}}, "costPerInputToken": 0, "costPerOutputToken": 2}`)
	limit := int64(10)
	req := request("capped", 0, "c1", 0)
	req.Body = &fwkrequest.InferenceRequestBody{MaxOutputTokens: &limit}

	require.Equal(t, "capped", pickID(t, p, band(&minVirtualService{}, queueOf(req, clk.t))))
	assert.InDelta(t, 1000.0-20.0, balance(t, p, "capped"), 1e-9)
}

func TestRate_OneBucketAcrossPriorities(t *testing.T) {
	p, clk := newRatedPolicy(t, `{"flowRates": {"capped": {"rate": 1, "burst": 100}}, "costPerInputToken": 0, "costPerOutputToken": 1}`)

	require.Equal(t, "capped", pickID(t, p, band(&minVirtualService{}, queueOf(request("capped", 0, "c1", 0), clk.t))))
	assert.Empty(t, pickID(t, p, band(&minVirtualService{}, queueOf(request("capped", 10, "c2", 0), clk.t))))
}

func TestRate_LongRunThroughputMatchesRate(t *testing.T) {
	const (
		rate      = 100.0
		burst     = 200.0
		reqCost   = 50
		seconds   = 60
		cycleStep = 10 * time.Millisecond
	)
	p, clk := newRatedPolicy(t, fmt.Sprintf(
		`{"flowRates": {"capped": {"rate": %v, "burst": %v}}, "costPerInputToken": 0, "costPerOutputToken": 1}`, rate, burst))
	state := &minVirtualService{}

	dispatched := 0
	for i := range seconds * int(time.Second/cycleStep) {
		req := request("capped", 0, fmt.Sprintf("c%d", i), 0)
		if pickID(t, p, band(state, queueOf(req, clk.t))) == "capped" {
			finish(p, req, 0, 0, reqCost)
			dispatched += reqCost
		}
		clk.advance(cycleStep)
	}
	assert.GreaterOrEqual(t, float64(dispatched), rate*seconds)
	assert.LessOrEqual(t, float64(dispatched), rate*seconds+burst+reqCost)
}

func TestRate_UnlimitedFlowsKeepNoRateState(t *testing.T) {
	p, clk := newRatedPolicy(t, `{"flowRates": {"capped": {"rate": 1}}}`)
	req := request("free", 0, "f1", 400)

	require.Equal(t, "free", pickID(t, p, band(&minVirtualService{}, queueOf(req, clk.t))))
	finish(p, req, 100, 0, 10)

	_, hasBucket := p.rates.Load("free")
	_, hasPending := p.pending.Load("f1")
	assert.False(t, hasBucket)
	assert.False(t, hasPending)
	assert.InDelta(t, 120.0, virtualService(p, "free"), 1e-9)
}

func TestPick_RequestWithoutIDChargedAtEndOnly(t *testing.T) {
	p, clk := newRatedPolicy(t, `{"flowRates": {"capped": {"rate": 1, "burst": 1000}}, "costPerInputToken": 0, "costPerOutputToken": 1}`)
	req := request("capped", 0, "", 0)

	require.Equal(t, "capped", pickID(t, p, band(&minVirtualService{}, queueOf(req, clk.t))))
	assert.InDelta(t, 1000.0, balance(t, p, "capped"), 1e-9)
	finish(p, req, 0, 0, 30)
	assert.InDelta(t, 970.0, balance(t, p, "capped"), 1e-9)
}

func TestPrune_RateState(t *testing.T) {
	p, clk := newRatedPolicy(t, `{"flowRates": {"idle": {"rate": 10, "burst": 10}, "indebt": {"rate": 0.001, "burst": 10}},
		"costPerInputToken": 0, "costPerOutputToken": 1, "idleTtlSeconds": 60, "sweepSeconds": 1, "pendingTtlSeconds": 30}`)
	state := &minVirtualService{}
	require.Equal(t, "idle", pickID(t, p, band(state, queueOf(request("idle", 0, "i1", 0), clk.t))))
	require.Equal(t, "indebt", pickID(t, p, band(state, queueOf(request("indebt", 0, "d1", 0), clk.t))))

	clk.advance(120 * time.Second)
	p.maybePrune(clk.t)

	_, idleKept := p.rates.Load("idle")
	_, indebtKept := p.rates.Load("indebt")
	_, pendingKept := p.pending.Load("i1")
	assert.False(t, idleKept, "idle bucket refilled to full is dropped")
	assert.True(t, indebtKept, "bucket still below burst keeps its debt")
	assert.Less(t, balance(t, p, "indebt"), 0.0)
	assert.False(t, pendingKept, "estimate past the pending TTL is dropped")
}

func TestRateMetrics(t *testing.T) {
	p, clk := newRatedPolicy(t, `{"flowRates": {"capped": {"rate": 5, "burst": 100}}, "costPerInputToken": 0, "costPerOutputToken": 1}`)
	req := request("capped", 0, "c1", 0)

	require.Equal(t, "capped", pickID(t, p, band(&minVirtualService{}, queueOf(req, clk.t))))
	finish(p, req, 0, 0, 30)

	assert.InDelta(t, 5.0, testutil.ToFloat64(p.rateGauge.WithLabelValues("capped")), 1e-9)
	assert.InDelta(t, 70.0, testutil.ToFloat64(p.balanceGauge.WithLabelValues("capped")), 1e-9)
	assert.InDelta(t, 30.0, testutil.ToFloat64(p.costCounter.WithLabelValues("capped")), 1e-9)
}

func TestPick_HeldRequestDispatchesAfterMaxHold(t *testing.T) {
	p, clk := newRatedPolicy(t, `{"flowRates": {"capped": {"rate": 1, "burst": 10}}, "maxHoldSeconds": 120,
		"costPerInputToken": 0, "costPerOutputToken": 1}`)
	state := &minVirtualService{}
	require.Equal(t, "capped", pickID(t, p, band(state, queueOf(request("capped", 0, "c1", 0), clk.t))))
	require.Negative(t, balance(t, p, "capped"))

	assert.Empty(t, pickID(t, p, band(state, queueOf(request("capped", 0, "c2", 0), clk.t.Add(-119*time.Second)))))
	assert.Equal(t, "capped", pickID(t, p, band(state, queueOf(request("capped", 0, "c2", 0), clk.t.Add(-120*time.Second)))))
}

func TestRate_DebtFlooredAtMaxHoldOfRefill(t *testing.T) {
	p, clk := newRatedPolicy(t, `{"flowRates": {"capped": {"rate": 10, "burst": 100}}, "maxHoldSeconds": 5,
		"costPerInputToken": 0, "costPerOutputToken": 1}`)
	finish(p, request("capped", 0, "c1", 0), 0, 0, 1000)
	assert.InDelta(t, -50.0, balance(t, p, "capped"), 1e-9)

	// A request arriving at the floor is eligible within maxHoldSeconds of refill.
	state := &minVirtualService{}
	clk.advance(5 * time.Second)
	assert.Empty(t, pickID(t, p, band(state, queueOf(request("capped", 0, "c2", 0), clk.t))))
	clk.advance(100 * time.Millisecond)
	assert.Equal(t, "capped", pickID(t, p, band(state, queueOf(request("capped", 0, "c2", 0), clk.t))))
}

func TestSettle_RefundsOnlyWhatTheFloorLetThrough(t *testing.T) {
	// Floor -50. The 256 estimate debits 150 (100 -> -50); settling to an actual cost of 30
	// leaves 100 - 30 = 70.
	p, clk := newRatedPolicy(t, `{"flowRates": {"capped": {"rate": 10, "burst": 100}}, "maxHoldSeconds": 5,
		"costPerInputToken": 0, "costPerOutputToken": 1}`)
	req := request("capped", 0, "c1", 0)
	require.Equal(t, "capped", pickID(t, p, band(&minVirtualService{}, queueOf(req, clk.t))))
	require.InDelta(t, -50.0, balance(t, p, "capped"), 1e-9)

	finish(p, req, 0, 0, 30)
	assert.InDelta(t, 70.0, balance(t, p, "capped"), 1e-9)
}
