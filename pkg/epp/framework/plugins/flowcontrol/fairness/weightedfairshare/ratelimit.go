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

// Rate ceilings: a token bucket per fairness ID, in the policy's cost units.
//
// Weights apply only while a band is contended. A ceiling applies at all times. Each fairness ID
// named in FlowRates, or every ID through DefaultFlowRate, gets a bucket that refills at Rate cost
// units per second up to Burst. Pick skips a flow whose balance is not positive, so its requests
// wait in their queue, subject to the request TTL, while other flows dispatch. When no flow in the
// band is eligible, Pick returns nil and the band waits for the next dispatch cycle.
//
// The bucket is keyed by fairness ID alone, so one ceiling spans every priority the ID sends at.
//
// Cost is known only at end of stream. Charging only then would let a flow dispatch any number of
// requests before the first of them is charged, so Pick charges an estimate for the request it
// selects and ResponseBody charges the difference to the actual cost. The estimate is learned per
// fairness ID from its completed requests: input cost per request byte and output cost per request.
// The balance may go negative, so a request larger than the burst dispatches once the balance is
// positive. Over time a flow dispatches its rate, exceeding it by at most the burst plus one
// request.
//
// With MaxHoldSeconds set, a held request dispatches once it has waited that long, so a ceiling
// delays requests and never causes one to expire in queue. The balance is then floored at
// -Rate*MaxHoldSeconds: a request arriving at the floor waits at most MaxHoldSeconds for the
// refill, and a sustained overload does not leave a debt that holds the flow after it ends.

import (
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// FlowRate is a token-bucket ceiling in cost units.
type FlowRate struct {
	// Rate is the refill in cost units per second. Zero means no ceiling.
	Rate float64 `json:"rate,omitempty"`
	// Burst is the bucket size in cost units. Zero takes defaultBurstSeconds of Rate.
	Burst float64 `json:"burst,omitempty"`
}

const (
	defaultBurstSeconds = 10
	// estimateAlpha is the weight of the newest completed request in a flow's cost estimates.
	estimateAlpha = 0.2
)

func (r FlowRate) limited() bool { return r.Rate > 0 }

func (r FlowRate) validate(field string) error {
	switch {
	case r.Rate < 0:
		return fmt.Errorf("%s: rate must be >= 0, got %v", field, r.Rate)
	case r.Burst < 0:
		return fmt.Errorf("%s: burst must be >= 0, got %v", field, r.Burst)
	case r.Rate == 0 && r.Burst > 0:
		return fmt.Errorf("%s: burst %v set without a rate", field, r.Burst)
	}
	return nil
}

func (r FlowRate) withDefaults() FlowRate {
	if r.Rate > 0 && r.Burst == 0 {
		r.Burst = r.Rate * defaultBurstSeconds
	}
	return r
}

// rateState is one fairness ID's bucket and cost estimates.
type rateState struct {
	mu      sync.Mutex
	rate    FlowRate
	balance float64
	// refilled is when balance was last brought up to date.
	refilled time.Time

	// Learned cost estimates. Seeded from the config; replaced by the first completed request,
	// then moved by estimateAlpha per completion.
	inCostPerByte float64
	inCostPerReq  float64
	outCost       float64
	samples       int
}

// debit subtracts cost from the balance, down to floor. Caller holds mu.
func (s *rateState) debit(cost, floor float64) {
	s.balance = math.Max(floor, s.balance-cost)
}

// refill brings the balance up to now and returns it. Caller holds mu.
func (s *rateState) refill(now time.Time) float64 {
	if elapsed := now.Sub(s.refilled).Seconds(); elapsed > 0 {
		s.balance = math.Min(s.rate.Burst, s.balance+s.rate.Rate*elapsed)
		s.refilled = now
	}
	return s.balance
}

// estimate is the expected cost of a request of the given size. Caller holds mu.
func (s *rateState) estimate(cfg Config, bytes int, maxOutputTokens *int64) float64 {
	in := s.inCostPerReq
	if bytes > 0 {
		in = s.inCostPerByte * float64(bytes)
	}
	out := s.outCost
	if maxOutputTokens != nil && *maxOutputTokens >= 0 {
		out = math.Min(out, cfg.CostPerOutputToken*float64(*maxOutputTokens))
	}
	return cfg.CostPerRequest + in + out
}

// learn folds a completed request's input and output cost into the estimates. Caller holds mu.
func (s *rateState) learn(bytes int, inputCost, outputCost float64) {
	alpha := estimateAlpha
	if s.samples == 0 {
		alpha = 1
	}
	s.samples++
	if bytes > 0 {
		s.inCostPerByte += alpha * (inputCost/float64(bytes) - s.inCostPerByte)
	}
	s.inCostPerReq += alpha * (inputCost - s.inCostPerReq)
	s.outCost += alpha * (outputCost - s.outCost)
}

// pendingCharge is the part of a dispatch estimate debited from the balance, not yet settled at
// end of stream.
type pendingCharge struct {
	amount float64
	at     time.Time
}

func (p *weightedFairShare) rateFor(id string) FlowRate {
	if r, ok := p.cfg.FlowRates[id]; ok {
		return r
	}
	return p.cfg.DefaultFlowRate
}

// rateStateFor returns the fairness ID's bucket, creating it full, or nil when the ID has no ceiling.
func (p *weightedFairShare) rateStateFor(id string, now time.Time) *rateState {
	if v, ok := p.rates.Load(id); ok {
		if st, ok := v.(*rateState); ok {
			return st
		}
	}
	rate := p.rateFor(id)
	if !rate.limited() {
		return nil
	}
	candidate := &rateState{
		rate:          rate,
		balance:       rate.Burst,
		refilled:      now,
		inCostPerByte: p.cfg.CostPerInputToken / p.cfg.BytesPerInputToken,
		outCost:       p.cfg.CostPerOutputToken * p.cfg.InitialOutputTokens,
	}
	actual, loaded := p.rates.LoadOrStore(id, candidate)
	st, ok := actual.(*rateState)
	if !ok {
		return candidate
	}
	if !loaded {
		setIDGauge(p.rateGauge, id, rate.Rate)
		setIDGauge(p.balanceGauge, id, rate.Burst)
	}
	return st
}

// rateAllows reports whether the fairness ID may dispatch now: it has no ceiling, or its bucket
// holds a positive balance.
func (p *weightedFairShare) rateAllows(id string, now time.Time) bool {
	st := p.rateStateFor(id, now)
	if st == nil {
		return true
	}
	st.mu.Lock()
	balance := st.refill(now)
	st.mu.Unlock()
	setIDGauge(p.balanceGauge, id, balance)
	return balance > 0
}

// heldLongEnough reports whether a request enqueued at enqueued has waited MaxHoldSeconds.
func (p *weightedFairShare) heldLongEnough(enqueued, now time.Time) bool {
	return p.cfg.MaxHoldSeconds > 0 && now.Sub(enqueued).Seconds() >= p.cfg.MaxHoldSeconds
}

// balanceFloor is the lowest balance a debit can reach: -Rate*MaxHoldSeconds, or unbounded
// without MaxHoldSeconds.
func (p *weightedFairShare) balanceFloor(rate FlowRate) float64 {
	if p.cfg.MaxHoldSeconds <= 0 {
		return math.Inf(-1)
	}
	return -rate.Rate * p.cfg.MaxHoldSeconds
}

// chargeEstimate charges the expected cost of the request at the head of the picked queue. A
// request picked again after a dispatch that did not go through is charged once.
func (p *weightedFairShare) chargeEstimate(queue flowcontrol.FlowQueueAccessor, now time.Time) {
	head := queue.Peek()
	if head == nil {
		return
	}
	id := queue.FlowKey().ID
	st := p.rateStateFor(id, now)
	if st == nil {
		return
	}
	fcReq := head.OriginalRequest()
	if fcReq == nil {
		return
	}
	ir := fcReq.InferenceRequest()
	// ResponseBody settles by RequestID; a request without one is charged at end of stream only.
	if ir == nil || ir.RequestID == "" {
		return
	}

	st.mu.Lock()
	if _, loaded := p.pending.Load(ir.RequestID); loaded {
		st.mu.Unlock()
		return
	}
	est := st.estimate(p.cfg, requestBytes(ir, fcReq.ByteSize()), maxOutputTokens(ir))
	before := st.refill(now)
	st.debit(est, p.balanceFloor(st.rate))
	balance := st.balance
	// The floor can absorb part of the estimate; settling refunds only what was debited.
	p.pending.Store(ir.RequestID, pendingCharge{amount: before - balance, at: now})
	st.mu.Unlock()
	setIDGauge(p.balanceGauge, id, balance)
}

// settleRate charges the difference between a completed request's actual cost and the estimate
// charged at dispatch, and updates the fairness ID's estimates. A request with no pending
// estimate (dispatched before this policy instance started) is charged its full cost.
func (p *weightedFairShare) settleRate(
	id string,
	request *fwksched.InferenceRequest,
	inputCost, outputCost, cost float64,
	now time.Time,
) {
	st := p.rateStateFor(id, now)
	if st == nil {
		return
	}
	charged := 0.0
	if request.RequestID != "" {
		if v, ok := p.pending.LoadAndDelete(request.RequestID); ok {
			if pc, ok := v.(pendingCharge); ok {
				charged = pc.amount
			}
		}
	}
	st.mu.Lock()
	st.refill(now)
	st.debit(cost-charged, p.balanceFloor(st.rate))
	st.learn(requestBytes(request, 0), inputCost, outputCost)
	balance := st.balance
	st.mu.Unlock()
	setIDGauge(p.balanceGauge, id, balance)
}

// prunePending drops dispatch estimates older than the pending TTL. Their requests reported no end
// of stream, and the estimate stays charged.
func (p *weightedFairShare) prunePending(now time.Time) {
	pendingTTL := time.Duration(p.cfg.PendingTTLSeconds * float64(time.Second))
	p.pending.Range(func(key, value any) bool {
		if pc, ok := value.(pendingCharge); !ok || now.Sub(pc.at) > pendingTTL {
			p.pending.Delete(key)
		}
		return true
	})
}

// pruneRates drops buckets idle beyond the idle TTL that have refilled to full, so a pruned ID
// returns to the same full bucket it would have had.
func (p *weightedFairShare) pruneRates(now time.Time) {
	idleTTL := time.Duration(p.cfg.IdleTTLSeconds * float64(time.Second))
	p.rates.Range(func(key, value any) bool {
		st, ok := value.(*rateState)
		if !ok {
			p.rates.Delete(key)
			return true
		}
		st.mu.Lock()
		idle := now.Sub(st.refilled) > idleTTL
		full := st.refill(now) >= st.rate.Burst
		st.mu.Unlock()
		if idle && full {
			p.rates.Delete(key)
			if id, ok := key.(string); ok {
				p.rateGauge.DeleteLabelValues(id)
				p.balanceGauge.DeleteLabelValues(id)
			}
		}
		return true
	})
}

func requestBytes(ir *fwksched.InferenceRequest, fallback uint64) int {
	if ir != nil && ir.RequestSizeBytes > 0 {
		return ir.RequestSizeBytes
	}
	if fallback > 0 && fallback <= math.MaxInt32 {
		return int(fallback)
	}
	return 0
}

func maxOutputTokens(ir *fwksched.InferenceRequest) *int64 {
	if ir == nil || ir.Body == nil {
		return nil
	}
	return ir.Body.MaxOutputTokens
}
