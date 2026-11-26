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
// Package weightedfairshare implements a fairness policy that shares a priority band between
// flows in proportion to operator-configured weights.
package weightedfairshare

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// WeightedFairShareFairnessPolicyType is the registration type for this policy.
const WeightedFairShareFairnessPolicyType = "weighted-fair-share-fairness-policy"

// Config is the JSON parameters block for the policy.
type Config struct {
	IdleTTLSeconds float64 `json:"idleTtlSeconds,omitempty"`
	SweepSeconds   float64 `json:"sweepSeconds,omitempty"`

	// FlowWeights maps fairness ID to weight. Requests carry no weight.
	FlowWeights map[string]float64 `json:"flowWeights,omitempty"`

	// DefaultFlowWeight applies to any fairness ID absent from FlowWeights.
	DefaultFlowWeight float64 `json:"defaultFlowWeight,omitempty"`

	// Cost of a completed request, the quantity whose share follows the weights:
	//
	//	cost = CostPerRequest + CostPerInputToken*uncachedPromptTokens
	//	     + CostPerCachedInputToken*cachedPromptTokens + CostPerOutputToken*completionTokens
	//
	// cachedPromptTokens is the model server's usage.prompt_tokens_details.cached_tokens.
	CostPerRequest     float64 `json:"costPerRequest,omitempty"`
	CostPerInputToken  float64 `json:"costPerInputToken,omitempty"`
	CostPerOutputToken float64 `json:"costPerOutputToken,omitempty"`
	// CostPerCachedInputToken defaults to CostPerInputToken when unset.
	CostPerCachedInputToken *float64 `json:"costPerCachedInputToken,omitempty"`

	// FlowRates maps fairness ID to a ceiling on cost per second; see ratelimit.go.
	FlowRates map[string]FlowRate `json:"flowRates,omitempty"`
	// DefaultFlowRate applies to any fairness ID absent from FlowRates. The zero value is no ceiling.
	DefaultFlowRate FlowRate `json:"defaultFlowRate,omitempty"`
	// InitialOutputTokens and BytesPerInputToken seed a rate-limited flow's dispatch estimate
	// until its first request completes.
	InitialOutputTokens float64 `json:"initialOutputTokens,omitempty"`
	BytesPerInputToken  float64 `json:"bytesPerInputToken,omitempty"`
	// PendingTTLSeconds bounds how long a dispatch estimate waits for its request's end of stream.
	PendingTTLSeconds float64 `json:"pendingTtlSeconds,omitempty"`
	// MaxHoldSeconds bounds how long a ceiling holds a request: a request of a flow over its ceiling
	// dispatches once it has waited this long. Zero holds it until the request TTL.
	MaxHoldSeconds float64 `json:"maxHoldSeconds,omitempty"`
}

// DefaultConfig returns the configuration used when no parameters are supplied.
func DefaultConfig() Config {
	return Config{
		IdleTTLSeconds:      3600,
		SweepSeconds:        300,
		DefaultFlowWeight:   1,
		CostPerRequest:      0,
		CostPerInputToken:   1,
		CostPerOutputToken:  2,
		InitialOutputTokens: 256,
		BytesPerInputToken:  4,
		PendingTTLSeconds:   3600,
	}
}

func (c Config) cachedInputPrice() float64 {
	if c.CostPerCachedInputToken != nil {
		return *c.CostPerCachedInputToken
	}
	return c.CostPerInputToken
}

func (c Config) validate() error {
	switch {
	case c.IdleTTLSeconds < 0:
		return fmt.Errorf("idleTtlSeconds must be >= 0, got %v", c.IdleTTLSeconds)
	case c.SweepSeconds <= 0:
		return fmt.Errorf("sweepSeconds must be > 0, got %v", c.SweepSeconds)
	case c.CostPerRequest < 0:
		return fmt.Errorf("costPerRequest must be >= 0, got %v", c.CostPerRequest)
	case c.CostPerInputToken < 0:
		return fmt.Errorf("costPerInputToken must be >= 0, got %v", c.CostPerInputToken)
	case c.CostPerOutputToken < 0:
		return fmt.Errorf("costPerOutputToken must be >= 0, got %v", c.CostPerOutputToken)
	case c.CostPerRequest == 0 && c.CostPerInputToken == 0 && c.CostPerOutputToken == 0:
		return errors.New("cost coefficients cannot all be zero")
	case c.DefaultFlowWeight <= 0:
		return fmt.Errorf("defaultFlowWeight must be > 0, got %v", c.DefaultFlowWeight)
	case c.CostPerCachedInputToken != nil && *c.CostPerCachedInputToken < 0:
		return fmt.Errorf("costPerCachedInputToken must be >= 0, got %v", *c.CostPerCachedInputToken)
	case c.InitialOutputTokens < 0:
		return fmt.Errorf("initialOutputTokens must be >= 0, got %v", c.InitialOutputTokens)
	case c.BytesPerInputToken <= 0:
		return fmt.Errorf("bytesPerInputToken must be > 0, got %v", c.BytesPerInputToken)
	case c.PendingTTLSeconds <= 0:
		return fmt.Errorf("pendingTtlSeconds must be > 0, got %v", c.PendingTTLSeconds)
	case c.MaxHoldSeconds < 0:
		return fmt.Errorf("maxHoldSeconds must be >= 0, got %v", c.MaxHoldSeconds)
	}
	for id, w := range c.FlowWeights {
		if w <= 0 {
			return fmt.Errorf("flowWeights[%q]: weight must be > 0, got %v", id, w)
		}
	}
	if err := c.DefaultFlowRate.validate("defaultFlowRate"); err != nil {
		return err
	}
	for id, r := range c.FlowRates {
		if err := r.validate(fmt.Sprintf("flowRates[%q]", id)); err != nil {
			return err
		}
	}
	return nil
}

// withRateDefaults fills each ceiling's unset burst.
func (c Config) withRateDefaults() Config {
	c.DefaultFlowRate = c.DefaultFlowRate.withDefaults()
	if len(c.FlowRates) > 0 {
		rates := make(map[string]FlowRate, len(c.FlowRates))
		for id, r := range c.FlowRates {
			rates[id] = r.withDefaults()
		}
		c.FlowRates = rates
	}
	return c
}

var (
	_ flowcontrol.FairnessPolicy  = &weightedFairShare{}
	_ fwkrh.ResponseBodyProcessor = &weightedFairShare{}
)

type weightedFairShare struct {
	name string
	cfg  Config

	// Keyed by FlowKey, so a tenant's flows at different priorities are accounted separately.
	state sync.Map // flowcontrol.FlowKey -> *flowState

	// Keyed by fairness ID: a ceiling spans every priority the ID sends at.
	rates sync.Map // string -> *rateState
	// Dispatch estimates awaiting end of stream, keyed by request ID.
	pending sync.Map // string -> pendingCharge

	// Unix-nano time of the last idle sweep, run from Pick.
	lastSweep atomic.Int64

	now func() time.Time

	// Per-instance so that several instances can share one registry.
	vserviceGauge *prometheus.GaugeVec
	weightGauge   *prometheus.GaugeVec
	rateGauge     *prometheus.GaugeVec
	balanceGauge  *prometheus.GaugeVec
	costCounter   *prometheus.CounterVec
}

// WeightedFairShareFairnessPolicyFactory constructs the policy from its parameters block.
func WeightedFairShareFairnessPolicyFactory(
	name string,
	parameters *json.Decoder,
	handle fwkplugin.Handle,
) (fwkplugin.Plugin, error) {
	cfg := DefaultConfig()
	if parameters != nil {
		if err := parameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("invalid config for %s plugin %q: %w", WeightedFairShareFairnessPolicyType, name, err)
		}
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", WeightedFairShareFairnessPolicyType, name, err)
	}
	if name == "" {
		name = WeightedFairShareFairnessPolicyType
	}
	cfg = cfg.withRateDefaults()

	vservice, weight := newCollectors(name)
	rate, balance, cost := newRateCollectors(name)
	if handle != nil {
		if reg := handle.Metrics(); reg != nil {
			for _, c := range []prometheus.Collector{vservice, weight, rate, balance, cost} {
				if err := reg.Register(c); err != nil {
					return nil, fmt.Errorf("%s plugin %q: registering metrics: %w",
						WeightedFairShareFairnessPolicyType, name, err)
				}
			}
		}
	}
	// A fairness ID the operator gave a weight or a ceiling keeps its own metric label after the label cap fills.
	for id := range cfg.FlowWeights {
		metricsutil.PinFairnessID(id)
	}
	for id := range cfg.FlowRates {
		metricsutil.PinFairnessID(id)
	}
	return &weightedFairShare{
		name: name, cfg: cfg, now: time.Now,
		vserviceGauge: vservice, weightGauge: weight,
		rateGauge: rate, balanceGauge: balance, costCounter: cost,
	}, nil
}

func (p *weightedFairShare) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: WeightedFairShareFairnessPolicyType, Name: p.name}
}

// minVirtualService is the virtual service of the flow a band last picked. It never decreases.
type minVirtualService struct {
	mu             sync.Mutex
	virtualService float64
}

// NewState returns the band's minVirtualService.
func (p *weightedFairShare) NewState(_ context.Context) any { return &minVirtualService{} }

// Pick serves the flow with the smallest virtual service (charged cost divided by weight), with
// head wait as the tie-break. Candidates below the band's minVirtualService are first raised to
// it, so a flow that was idle or is new gets no credit for the time it was absent.
func (p *weightedFairShare) Pick(
	_ context.Context,
	band flowcontrol.PriorityBandAccessor,
) (flowcontrol.FlowQueueAccessor, error) {
	if band == nil {
		return nil, nil //nolint:nilnil
	}
	minVS, ok := band.PolicyState().(*minVirtualService)
	if !ok || minVS == nil {
		return nil, fmt.Errorf("invalid state type for %s policy: expected *minVirtualService, got %T",
			WeightedFairShareFairnessPolicyType, band.PolicyState())
	}
	minVS.mu.Lock()
	defer minVS.mu.Unlock()

	now := p.now()
	var best flowcontrol.FlowQueueAccessor
	bestVS, bestWait := math.Inf(1), time.Duration(math.MinInt64)

	band.IterateQueues(func(queue flowcontrol.FlowQueueAccessor) bool {
		if queue == nil || queue.Len() == 0 {
			return true
		}
		head := queue.Peek()
		if head == nil {
			return true
		}
		key := queue.FlowKey()
		// A flow over its ceiling is skipped before catchUp, so it is raised to the band's
		// minimum when it becomes eligible and earns no credit for the time it was held.
		if !p.rateAllows(key.ID, now) && !p.heldLongEnough(head.EnqueueTime(), now) {
			return true
		}
		vs, raised := p.stateFor(key).catchUp(minVS.virtualService, now)
		if raised {
			setFlowGauge(p.vserviceGauge, key, vs)
		}

		wait := now.Sub(head.EnqueueTime())
		if vs < bestVS || (vs == bestVS && wait > bestWait) {
			best, bestVS, bestWait = queue, vs, wait
		}
		return true
	})

	p.maybePrune(now)
	if best != nil {
		minVS.virtualService = max(minVS.virtualService, bestVS)
		p.chargeEstimate(best, now)
	}
	return best, nil
}

// ResponseBody charges the flow for a completed request.
func (p *weightedFairShare) ResponseBody(
	_ context.Context,
	request *fwksched.InferenceRequest,
	response *fwkrh.Response,
	_ *datalayer.EndpointMetadata,
) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	promptTokens := response.Usage.PromptTokens
	completionTokens := response.Usage.CompletionTokens
	if completionTokens <= 0 && response.StreamedEvents > 0 {
		completionTokens = response.StreamedEvents
		// The token producer runs after flow control, so the policy cannot declare
		// TokenizedPrompt as a dependency. At end of stream the body carries its count when one ran.
		if promptTokens <= 0 && request.Body != nil {
			promptTokens = request.Body.TokenizedRequest.TokenCount()
		}
	}
	cachedTokens := 0
	if details := response.Usage.PromptTokenDetails; details != nil {
		cachedTokens = min(max(details.CachedTokens, 0), max(promptTokens, 0))
	}
	inputCost := p.cfg.CostPerInputToken*float64(promptTokens-cachedTokens) +
		p.cfg.cachedInputPrice()*float64(cachedTokens)
	outputCost := p.cfg.CostPerOutputToken * float64(completionTokens)
	cost := p.cfg.CostPerRequest + inputCost + outputCost

	id := fairnessIDFor(request)
	now := p.now()
	p.settleRate(id, request, inputCost, outputCost, cost, now)
	if cost == 0 {
		return
	}
	if label, overflow := metricsutil.BoundFairnessIDWithOverflow(id); !overflow {
		p.costCounter.WithLabelValues(label).Add(cost)
	}
	key := flowcontrol.FlowKey{ID: id, Priority: request.Objectives.Priority}
	vs := p.stateFor(key).charge(cost/p.weightFor(key), now)
	setFlowGauge(p.vserviceGauge, key, vs)
}

// flowState is one flow's virtual service; lastSeen drives idle pruning.
type flowState struct {
	mu             sync.Mutex
	virtualService float64
	lastSeen       time.Time
}

// charge adds delta and returns the new virtual service.
func (s *flowState) charge(delta float64, now time.Time) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.virtualService += delta
	s.lastSeen = now
	return s.virtualService
}

// catchUp raises virtual service to at least minVS and returns it, and whether it was raised.
func (s *flowState) catchUp(minVS float64, now time.Time) (float64, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastSeen = now
	if s.virtualService >= minVS {
		return s.virtualService, false
	}
	s.virtualService = minVS
	return minVS, true
}

func (s *flowState) lastTouch() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastSeen
}

func (p *weightedFairShare) stateFor(key flowcontrol.FlowKey) *flowState {
	if v, ok := p.state.Load(key); ok {
		if st, ok := v.(*flowState); ok {
			return st
		}
	}
	candidate := &flowState{}
	actual, loaded := p.state.LoadOrStore(key, candidate)
	st, ok := actual.(*flowState)
	if !ok {
		return candidate
	}
	if !loaded {
		// The weight is fixed for the policy's lifetime, so it is published once per flow.
		setFlowGauge(p.weightGauge, key, p.weightFor(key))
		setFlowGauge(p.vserviceGauge, key, 0)
	}
	return st
}

// weightFor looks up the weight by fairness ID only, so it is the same at every priority.
func (p *weightedFairShare) weightFor(key flowcontrol.FlowKey) float64 {
	if w, ok := p.cfg.FlowWeights[key.ID]; ok && w > 0 {
		return w
	}
	return p.cfg.DefaultFlowWeight
}

// maybePrune runs an idle sweep at most once per sweepSeconds.
func (p *weightedFairShare) maybePrune(now time.Time) {
	interval := int64(p.cfg.SweepSeconds * float64(time.Second))
	last := p.lastSweep.Load()
	if last != 0 && now.UnixNano()-last < interval {
		return
	}
	if !p.lastSweep.CompareAndSwap(last, now.UnixNano()) {
		return // another caller owns this interval
	}
	p.prunePending(now)
	if p.cfg.IdleTTLSeconds <= 0 {
		return
	}
	p.prune(now)
	p.pruneRates(now)
}

// prune drops flows untouched for longer than the idle TTL, with their metric series.
func (p *weightedFairShare) prune(now time.Time) {
	ttl := time.Duration(p.cfg.IdleTTLSeconds * float64(time.Second))
	p.state.Range(func(key, value any) bool {
		st, ok := value.(*flowState)
		if !ok {
			p.state.Delete(key)
			return true
		}
		// A zero timestamp is a freshly created entry Pick has not read yet.
		if touched := st.lastTouch(); touched.IsZero() || now.Sub(touched) <= ttl {
			return true
		}
		p.state.Delete(key)
		if fk, ok := key.(flowcontrol.FlowKey); ok {
			priority := strconv.Itoa(fk.Priority)
			p.vserviceGauge.DeleteLabelValues(fk.ID, priority)
			p.weightGauge.DeleteLabelValues(fk.ID, priority)
		}
		return true
	})
}

// setFlowGauge sets a per-flow gauge under the shared fairness-ID label cap. Flows past the cap
// are skipped: one overflow series would only hold whichever flow wrote last.
func setFlowGauge(vec *prometheus.GaugeVec, key flowcontrol.FlowKey, v float64) {
	if label, overflow := metricsutil.BoundFairnessIDWithOverflow(key.ID); !overflow {
		vec.WithLabelValues(label, strconv.Itoa(key.Priority)).Set(v)
	}
}

func fairnessIDFor(req *fwksched.InferenceRequest) string {
	if req == nil || req.FairnessID == "" {
		return reqcommon.DefaultFairnessID
	}
	return req.FairnessID
}

func newCollectors(name string) (vservice, weight *prometheus.GaugeVec) {
	labels := prometheus.Labels{"policy": name}
	vservice = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Subsystem:   eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:        "weighted_fair_share_virtual_service",
		Help:        metricsutil.HelpMsgWithStability("Virtual service per fairness flow: charged cost divided by weight.", compbasemetrics.ALPHA),
		ConstLabels: labels,
	}, []string{"flow_id", "priority"})
	weight = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Subsystem:   eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:        "weighted_fair_share_flow_weight",
		Help:        metricsutil.HelpMsgWithStability("Weight resolved for a fairness flow.", compbasemetrics.ALPHA),
		ConstLabels: labels,
	}, []string{"flow_id", "priority"})
	return vservice, weight
}

func newRateCollectors(name string) (rate, balance *prometheus.GaugeVec, cost *prometheus.CounterVec) {
	labels := prometheus.Labels{"policy": name}
	rate = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Subsystem:   eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:        "weighted_fair_share_flow_rate",
		Help:        metricsutil.HelpMsgWithStability("Ceiling resolved for a rate-limited fairness ID, in cost per second.", compbasemetrics.ALPHA),
		ConstLabels: labels,
	}, []string{"flow_id"})
	balance = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Subsystem:   eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:        "weighted_fair_share_rate_balance",
		Help:        metricsutil.HelpMsgWithStability("Token-bucket balance of a rate-limited fairness ID, in cost; the ID dispatches only while it is positive.", compbasemetrics.ALPHA),
		ConstLabels: labels,
	}, []string{"flow_id"})
	cost = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem:   eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:        "weighted_fair_share_charged_cost_total",
		Help:        metricsutil.HelpMsgWithStability("Cost charged to a fairness ID at end of stream.", compbasemetrics.ALPHA),
		ConstLabels: labels,
	}, []string{"flow_id"})
	return rate, balance, cost
}

// setIDGauge sets a per-fairness-ID gauge under the shared fairness-ID label cap.
func setIDGauge(vec *prometheus.GaugeVec, id string, v float64) {
	if label, overflow := metricsutil.BoundFairnessIDWithOverflow(id); !overflow {
		vec.WithLabelValues(label).Set(v)
	}
}
