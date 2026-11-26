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

// A fairness ID over its rate ceiling is held in its queue until the request TTL while another
// fairness ID dispatches, and the ceiling is charged from the stream's usage block, with cached
// prompt tokens priced separately.
func TestWeightedFairShareRateCeiling(t *testing.T) {
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
      alpha: {rate: 0.001, burst: 10}
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
  defaultRequestTTL: "1s"
  priorityBands:
  - priority: 2
    fairnessPolicyRef: %[1]s
`, weightedfairshare.WeightedFairShareFairnessPolicyType)

	registry := crmetrics.Registry
	crmetrics.Registry = prometheus.NewRegistry()
	t.Cleanup(func() { crmetrics.Registry = registry })

	h := eppharness.NewTestHarness(t.Context(), t, eppharness.WithStandardMode(), eppharness.WithConfigText(configText)).WithBaseResources()
	pods := []eppharness.PodState{eppharness.P(0, 0, 0.1, modelMyModelTarget)}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel).WaitForMetricsDelivery()

	newClient := func() extProcPb.ExternalProcessor_ProcessClient {
		client, err := extProcPb.NewExternalProcessorClient(h.GRPCConn()).Process(t.Context())
		require.NoError(t, err)
		t.Cleanup(func() { _ = client.CloseSend() })
		return client
	}
	chat := func(fairnessID, requestID string) []*extProcPb.ProcessingRequest {
		return fwkepp.ReqRaw(map[string]string{
			":method":                    "POST",
			":path":                      "/v1/chat/completions",
			"content-type":               "application/json",
			reqcommon.RequestIDHeaderKey: requestID,
			metadata.ObjectiveKey:        modelMyModel,
			metadata.FlowFairnessIDKey:   fairnessID,
		}, fmt.Sprintf(`{"model":%q,"stream":true,"messages":[{"role":"user","content":"hello"}]}`, modelMyModel))
	}

	// 100 prompt tokens with 60 cached and 10 completion tokens cost 0.5*40 + 10 = 30, leaving
	// alpha's balance at 10 - 30 = -20.
	first := append(chat("alpha", "alpha-1"), ReqResponseOnly(
		map[string]string{"content-type": "text/event-stream", ":status": "200"},
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n",
		"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],"+
			"\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":10,\"total_tokens\":110,"+
			"\"prompt_tokens_details\":{\"cached_tokens\":60}}}\n\n",
		"data: [DONE]\n\n",
	)...)
	responses, err := fwkepp.StreamedRequest(t, newClient(), first, len(first))
	require.NoError(t, err)
	require.Len(t, responses, len(first))

	h.ExpectMetrics(map[string]string{
		"llm_d_epp_weighted_fair_share_charged_cost_total": fmt.Sprintf(`
# HELP llm_d_epp_weighted_fair_share_charged_cost_total [ALPHA] Cost charged to a fairness ID at end of stream.
# TYPE llm_d_epp_weighted_fair_share_charged_cost_total counter
llm_d_epp_weighted_fair_share_charged_cost_total{flow_id="alpha",policy="%s"} 30
`, weightedfairshare.WeightedFairShareFairnessPolicyType),
	})

	held := newClient()
	for _, req := range chat("alpha", "alpha-2") {
		require.NoError(t, held.Send(req))
	}

	responses, err = fwkepp.StreamedRequest(t, newClient(), chat("beta", "beta-1"), 2)
	require.NoError(t, err)
	require.Len(t, responses, 2, "beta has no ceiling and must be routed while alpha is held")
	for _, resp := range responses {
		assert.Nil(t, resp.GetImmediateResponse())
	}

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
