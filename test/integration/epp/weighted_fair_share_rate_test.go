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

package epp

import (
	"fmt"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	crmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/weightedfairshare"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
)

type rateCeilingEPP struct {
	newClient func() extProcPb.ExternalProcessor_ProcessClient
	chat      func(fairnessID, requestID string) []*extProcPb.ProcessingRequest
}

// startRateCeilingEPP runs an EPP whose only band uses weighted fair share with a ceiling on
// fairness ID alpha, then drives alpha's balance to 10 - 30 = -20 with one completed stream: 100
// prompt tokens with 60 cached and 10 completion tokens cost 0.5*40 + 10 = 30.
func startRateCeilingEPP(t *testing.T, extraParams, ttl string) rateCeilingEPP {
	t.Helper()
	configText := fmt.Sprintf(`
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
plugins:
- type: openai-parser
- type: mock-metrics-source
- type: random-picker
- type: %[1]s
  parameters:
    costPerInputToken: 0.5
    costPerCachedInputToken: 0
    costPerOutputToken: 1
    flowRates:
      alpha: {rate: 0.001, burst: 10}%[2]s
requestHandler:
  parsers:
  - pluginRef: openai-parser
dataLayer:
  injectDefaults: false
  sources:
  - pluginRef: mock-metrics-source
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: random-picker
flowControl:
  defaultRequestTTL: %[3]q
  priorityBands:
  - priority: 2
    fairnessPolicyRef: %[1]s
`, weightedfairshare.WeightedFairShareFairnessPolicyType, extraParams, ttl)

	registry := crmetrics.Registry
	crmetrics.Registry = prometheus.NewRegistry()
	t.Cleanup(func() { crmetrics.Registry = registry })

	h := eppharness.NewTestHarness(t.Context(), t, eppharness.WithStandardMode(), eppharness.WithConfigText(configText)).WithBaseResources()
	pods := []eppharness.PodState{eppharness.P(0, 0, 0.1, modelMyModelTarget)}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel).WaitForMetricsDelivery()

	e := rateCeilingEPP{
		newClient: func() extProcPb.ExternalProcessor_ProcessClient {
			client, err := extProcPb.NewExternalProcessorClient(h.GRPCConn()).Process(t.Context())
			require.NoError(t, err)
			t.Cleanup(func() { _ = client.CloseSend() })
			return client
		},
		chat: func(fairnessID, requestID string) []*extProcPb.ProcessingRequest {
			return fwkepp.ReqRaw(map[string]string{
				":method":                    "POST",
				":path":                      "/v1/chat/completions",
				"content-type":               "application/json",
				reqcommon.RequestIDHeaderKey: requestID,
				metadata.ObjectiveKey:        modelMyModel,
				metadata.FlowFairnessIDKey:   fairnessID,
			}, fmt.Sprintf(`{"model":%q,"stream":true,"messages":[{"role":"user","content":"hello"}]}`, modelMyModel))
		},
	}

	first := append(e.chat("alpha", "alpha-1"), ReqResponseOnly(
		map[string]string{"content-type": "text/event-stream", ":status": "200"},
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n",
		"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],"+
			"\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":10,\"total_tokens\":110,"+
			"\"prompt_tokens_details\":{\"cached_tokens\":60}}}\n\n",
		"data: [DONE]\n\n",
	)...)
	responses, err := fwkepp.StreamedRequest(t, e.newClient(), first, len(first))
	require.NoError(t, err)
	require.Len(t, responses, len(first))

	h.ExpectMetrics(map[string]string{
		"llm_d_epp_weighted_fair_share_charged_cost_total": fmt.Sprintf(`
# HELP llm_d_epp_weighted_fair_share_charged_cost_total [ALPHA] Cost charged to a fairness ID at end of stream.
# TYPE llm_d_epp_weighted_fair_share_charged_cost_total counter
llm_d_epp_weighted_fair_share_charged_cost_total{flow_id="alpha",policy="%s"} 30
`, weightedfairshare.WeightedFairShareFairnessPolicyType),
	})
	return e
}

// sendHeldAndRouteBeta sends alpha's next request, then requires beta, which has no ceiling, to be
// routed while alpha's request is held.
func (e rateCeilingEPP) sendHeldAndRouteBeta(t *testing.T) extProcPb.ExternalProcessor_ProcessClient {
	t.Helper()
	held := e.newClient()
	for _, req := range e.chat("alpha", "alpha-2") {
		require.NoError(t, held.Send(req))
	}
	responses, err := fwkepp.StreamedRequest(t, e.newClient(), e.chat("beta", "beta-1"), 2)
	require.NoError(t, err)
	require.Len(t, responses, 2, "beta must be routed while alpha is held")
	for _, resp := range responses {
		assert.Nil(t, resp.GetImmediateResponse())
	}
	return held
}

// Without maxHoldSeconds, a request over the ceiling stays queued until the request TTL.
func TestWeightedFairShareRateCeiling_HoldsUntilTTL(t *testing.T) {
	e := startRateCeilingEPP(t, "", "1s")
	held := e.sendHeldAndRouteBeta(t)

	resp, err := held.Recv()
	require.NoError(t, err)
	immediate := resp.GetImmediateResponse()
	require.NotNil(t, immediate, "alpha's request must stay queued until its TTL")
	assert.Equal(t, envoyTypePb.StatusCode_TooManyRequests, immediate.GetStatus().GetCode())
	reason := ""
	for _, hdr := range immediate.GetHeaders().GetSetHeaders() {
		if hdr.GetHeader().GetKey() == errcommon.RequestDroppedReasonHeaderKey {
			reason = string(hdr.GetHeader().GetRawValue())
		}
	}
	assert.Equal(t, string(errcommon.RequestDroppedReasonTTLExpired), reason)
}

// With maxHoldSeconds below the request TTL, a request over the ceiling is delayed and then
// routed. The balance is floored at -rate*maxHoldSeconds = -0.003, so the refill makes alpha
// eligible about 3 s after its first request settled.
func TestWeightedFairShareRateCeiling_MaxHoldDispatches(t *testing.T) {
	e := startRateCeilingEPP(t, "\n    maxHoldSeconds: 3", "10s")
	sent := time.Now()
	held := e.sendHeldAndRouteBeta(t)

	responses, err := fwkepp.StreamedRequest(t, held, nil, 2)
	require.NoError(t, err)
	require.Len(t, responses, 2, "alpha's request must be routed after maxHoldSeconds")
	for _, resp := range responses {
		assert.Nil(t, resp.GetImmediateResponse())
	}
	assert.GreaterOrEqual(t, time.Since(sent), time.Second)
}
