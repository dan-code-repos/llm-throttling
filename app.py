"""
LLM Model Throttler - Main Application
Streamlit interface for demonstrating production-grade model routing and throttling
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from core.models import ModelPool
from core.policy import RoutingPolicy
from core.throttler import ModelThrottler
from simulation.engine import SimulationEngine
from gateway.azure_gateway import AzureGatewayResponsibilities

# Page configuration
st.set_page_config(
    page_title="LLM Model Throttler",
    layout="wide"
)

# Initialize session state
if 'simulation_engine' not in st.session_state:
    model_pool = ModelPool()
    routing_policy = RoutingPolicy()
    throttler = ModelThrottler(model_pool, routing_policy)
    st.session_state.simulation_engine = SimulationEngine(throttler)
    st.session_state.decision_history = []
    st.session_state.event_log = []

# Main title
st.title(" Model Throttler demonstration")
st.markdown("**Production-Grade Model Routing Control Plane**")
st.divider()

# Sidebar - Simulation Controls
with st.sidebar:
    st.header("Simulation Controls")
    
    # Scenario selector
    scenario = st.selectbox(
        "Quick Scenarios",
        [
            "Manual Control",
            "Demo: Fallback Routing",
            "Demo: Queue Buildup",
            "Demo: Queue Draining"
        ],
        help="Select a scenario to automatically trigger specific behaviors"
    )
    
    st.divider()
    
    if scenario == "Manual Control":
        requests_per_step = st.slider(
            "Requests per Step",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of requests to generate in each simulation step"
        )
        
        completion_rate = st.slider(
            "Completion Rate %",
            min_value=0,
            max_value=100,
            value=30,
            help="Probability that an active request completes each step"
        )
    else:
        st.info(f"**Scenario Mode Active**\n\n{scenario}")
        if scenario == "Demo: Fallback Routing":
            requests_per_step = 8
            completion_rate = 20
            st.markdown("**Settings:**\n- 8 requests/step\n- 20% completion\n- Goal: Fill primary, trigger fallback")
        elif scenario == "Demo: Queue Buildup":
            requests_per_step = 15
            completion_rate = 10
            st.markdown("**Settings:**\n- 15 requests/step\n- 10% completion\n- Goal: Fill everything, build queues")
        elif scenario == "Demo: Queue Draining":
            requests_per_step = 2
            completion_rate = 80
            st.markdown("**Settings:**\n- 2 requests/step\n- 80% completion\n- Goal: Drain queues, watch recovery")
    
    # Update engine completion rate
    st.session_state.simulation_engine.completion_probability = completion_rate / 100.0
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        run_step = st.button("▶️ Run Step", use_container_width=True)
    with col2:
        reset_sim = st.button("🔄 Reset", use_container_width=True)
    
    if reset_sim:
        model_pool = ModelPool()
        routing_policy = RoutingPolicy()
        throttler = ModelThrottler(model_pool, routing_policy)
        st.session_state.simulation_engine = SimulationEngine(throttler)
        st.session_state.decision_history = []
        st.session_state.event_log = []
        st.rerun()
    
    st.divider()
    
    # Simulation stats
    engine = st.session_state.simulation_engine
    st.metric("Simulation Step", engine.current_step)
    st.metric("Total Requests", engine.total_requests)
    st.metric("Total Completed", engine.total_completed)
    
    # Event counters
    fallback_count = sum(1 for d in st.session_state.decision_history if d['decision'] == 'FALLBACK_ACCEPTED')
    queue_count = sum(1 for d in st.session_state.decision_history if d['decision'] == 'QUEUED')
    drain_count = len([e for e in st.session_state.event_log if 'DRAINED' in e])
    
    st.divider()
    st.markdown("### Key Events")
    st.metric("Fallback Used", fallback_count, delta="Last resort routing")
    st.metric("Requests Queued", queue_count, delta="Waiting for capacity")
    st.metric("Queue Drains", drain_count, delta="Auto-promoted")

# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Throttler State",
    "📋 Decision Trace",
    "🔔 Event Log",
    "🔧 Gateway Scope (Platform layer)",
    "ℹ️ About"
])

# Run simulation step if button clicked
if run_step:
    decisions, events = st.session_state.simulation_engine.run_step(requests_per_step)
    st.session_state.decision_history.extend(decisions)
    st.session_state.event_log.extend(events)
    # Keep only last 100 decisions and 100 events
    st.session_state.decision_history = st.session_state.decision_history[-100:]
    st.session_state.event_log = st.session_state.event_log[-100:]

# Tab 1: Throttler State Visualization
with tab1:
    st.header("Model Throttler State")
    st.markdown("Real-time view of per-model request states")
    
    throttler = st.session_state.simulation_engine.throttler
    
    # Prepare data for visualization
    state_data = []
    for model_name, model_info in throttler.model_pool.models.items():
        state = throttler.model_states[model_name]
        utilization = (state['active'] / model_info['max_concurrency'] * 100) if model_info['max_concurrency'] > 0 else 0
        state_data.append({
            'Model': model_name,
            'Role': model_info['role'],
            'Active (Executing)': state['active'],
            'Queued (Waiting)': state['queued'],
            'Max Concurrency': model_info['max_concurrency'],
            'Utilization %': f"{utilization:.0f}%",
            'Speed': model_info['speed'],
            'Accuracy': model_info['accuracy']
        })
    
    df_state = pd.DataFrame(state_data)
    
    # Highlight high utilization and queues
    def highlight_state(row):
        styles = [''] * len(row)
        if row['Queued (Waiting)'] > 0:
            styles[3] = 'background-color: #ffcccc'  # Highlight queued
        if row['Active (Executing)'] >= row['Max Concurrency']:
            styles[2] = 'background-color: #ffffcc'  # Highlight full
        return styles
    
    # Table view
    st.subheader("Detailed State Table")
    styled_df = df_state.style.apply(highlight_state, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Bar chart view
    st.subheader("Capacity Utilization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Active vs Capacity
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        x = range(len(df_state))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], df_state['Active (Executing)'], 
                width, label='Active (Executing)', color='#2ecc71')
        ax1.bar([i + width/2 for i in x], df_state['Max Concurrency'], 
                width, label='Max Concurrency', color='#95a5a6', alpha=0.5)
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Count')
        ax1.set_title('Active Requests vs Capacity')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_state['Model'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
    
    with col2:
        # Queued requests
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        colors = ['#e74c3c' if q > 0 else '#3498db' for q in df_state['Queued (Waiting)']]
        ax2.bar(df_state['Model'], df_state['Queued (Waiting)'], color=colors)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Count')
        ax2.set_title('Queued Requests (Waiting)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Status indicators
    st.divider()
    st.subheader("Model Status")
    cols = st.columns(len(state_data))
    for idx, model_data in enumerate(state_data):
        with cols[idx]:
            active = model_data['Active (Executing)']
            capacity = model_data['Max Concurrency']
            queued = model_data['Queued (Waiting)']
            
            if active >= capacity and queued > 0:
                status = "🔴 Full + Queue"
                status_color = "red"
            elif active >= capacity:
                status = "🟡 Full"
                status_color = "orange"
            elif active > 0:
                status = "🟢 Active"
                status_color = "green"
            else:
                status = "⚪ Idle"
                status_color = "gray"
            
            st.metric(
                model_data['Model'],
                f"{active}/{capacity}",
                delta=f"Queue: {queued}" if queued > 0 else "No queue",
                delta_color="inverse" if queued > 0 else "off"
            )
            st.caption(status)

# Tab 2: Decision Trace
with tab2:
    st.header("Request Decision Trace")
    st.markdown("Explainable routing decisions for the most recent requests")
    
    if st.session_state.decision_history:
        # Prepare decision history data
        trace_data = []
        for decision in reversed(st.session_state.decision_history[-50:]):  # Show last 50
            trace_data.append({
                'Step': decision['step'],
                'Request ID': decision['request_id'],
                'Intent': decision['intent'],
                'Primary': decision['primary_model'],
                'Selected Model': decision['selected_model'],
                'Decision': decision['decision'],
                'Reason': decision['reason']
            })
        
        df_trace = pd.DataFrame(trace_data)
        
        # Decision type filter
        decision_types = ['All'] + sorted(df_trace['Decision'].unique().tolist())
        selected_decision = st.selectbox("Filter by Decision Type", decision_types)
        
        if selected_decision != 'All':
            df_filtered = df_trace[df_trace['Decision'] == selected_decision]
        else:
            df_filtered = df_trace
        
        # Color-code decisions
        def highlight_decision(row):
            colors = {
                'PRIMARY_ACCEPTED': 'background-color: #d4edda',
                'FALLBACK_ACCEPTED': 'background-color: #fff3cd',
                'QUEUED': 'background-color: #f8d7da'
            }
            return [colors.get(row['Decision'], '')] * len(row)
        
        styled_df = df_filtered.style.apply(highlight_decision, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Decision statistics
        st.divider()
        st.subheader("📈 Decision Statistics")
        
        decision_counts = df_trace['Decision'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            primary = decision_counts.get('PRIMARY_ACCEPTED', 0)
            st.metric("✅ Primary Accepted", primary, 
                     delta=f"{primary/len(df_trace)*100:.1f}%" if len(df_trace) > 0 else "0%")
        with col2:
            fallback = decision_counts.get('FALLBACK_ACCEPTED', 0)
            st.metric("⚠️ Fallback Used", fallback,
                     delta=f"{fallback/len(df_trace)*100:.1f}%" if len(df_trace) > 0 else "0%",
                     delta_color="inverse")
        with col3:
            queued = decision_counts.get('QUEUED', 0)
            st.metric("⏸️ Queued", queued,
                     delta=f"{queued/len(df_trace)*100:.1f}%" if len(df_trace) > 0 else "0%",
                     delta_color="inverse")
        
    else:
        st.info("Run a simulation step to see decision traces")

# Tab 3: Event Log
with tab3:
    st.header("System Event Log")
    st.markdown("**Real-time log of key throttler events: fallbacks, queuing, and queue draining**")
    
    if st.session_state.event_log:
        # Show events in reverse chronological order
        st.markdown("### Recent Events")
        
        for event in reversed(st.session_state.event_log[-30:]):
            if 'FALLBACK' in event:
                st.warning(f"⚠️ {event}")
            elif 'QUEUED' in event:
                st.error(f"⏸️ {event}")
            elif 'DRAINED' in event:
                st.success(f"✅ {event}")
            elif 'COMPLETED' in event:
                st.info(f"✓ {event}")
            else:
                st.text(event)
        
        st.divider()
        
        # Event statistics
        st.subheader("Event Summary")
        
        fallback_events = [e for e in st.session_state.event_log if 'FALLBACK' in e]
        queue_events = [e for e in st.session_state.event_log if 'QUEUED' in e]
        drain_events = [e for e in st.session_state.event_log if 'DRAINED' in e]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fallback Events", len(fallback_events))
            if fallback_events:
                st.caption(f"Latest: {fallback_events[-1][:50]}...")
        
        with col2:
            st.metric("Queue Events", len(queue_events))
            if queue_events:
                st.caption(f"Latest: {queue_events[-1][:50]}...")
        
        with col3:
            st.metric("Drain Events", len(drain_events))
            if drain_events:
                st.caption(f"Latest: {drain_events[-1][:50]}...")
        
    else:
        st.info("Run a simulation step to see events")
        st.markdown("""
        **What you'll see here:**
        - 🟡 **Fallback routing** - When primary model is full
        - 🔴 **Queue additions** - When both primary and fallback are full
        - 🟢 **Queue draining** - When queued requests get promoted to active
        - 🔵 **Completions** - When active requests finish
        """)

# Tab 4: Gateway Responsibilities
with tab4:
    st.header("LLM Gateway Responsibilities")
    st.markdown("**These capabilities are handled by the platform layer (Azure OpenAI, API Gateway, etc.)**")
    
    st.warning("""
    **Note**
    
    The Model Throttler focuses on **policy-driven routing logic**. 
    The responsibilities mentioned below are **NOT implemented** in the throttler 
    but are **provided by platform tools**.
    """)
    
    azure_gateway = AzureGatewayResponsibilities()
    
    # Display platform responsibilities
    for category, items in azure_gateway.get_all_responsibilities().items():
        with st.expander(f"🔧 {category}", expanded=True):
            for item in items:
                st.markdown(f"- {item}")
    
    st.divider()
    
    # Comparison table
    st.subheader("Separation of Concerns")
    
    comparison_data = {
        'Capability': [
            'Intent-aware routing',
            'Primary/Fallback selection',
            'Queue management',
            'Business policy decisions',
            'Rate limiting (RPM/TPM)',
            'Token counting',
            'Retry logic',
            'Network security',
            'Connection pooling'
        ],
        'Model Throttler': [
            '✅ Yes', '✅ Yes', '✅ Yes', '✅ Yes',
            '❌ No', '❌ No', '❌ No', '❌ No', '❌ No'
        ],
        'Platform/Gateway': [
            '❌ No', '❌ No', '❌ No', '❌ No',
            '✅ Yes', '✅ Yes', '✅ Yes', '✅ Yes', '✅ Yes'
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

# Tab 5: About
with tab5:
    st.header("About")
    
    st.markdown("""
    ## Purpose
    
    This demonstrates a **production-grade model throttler / routing control plane** 
    for LLM-based systems, with clear separation from platform capabilities.
    
    ## How to See Each Feature
    
    ### 1️. Primary/Fallback Selection
    **Use Scenario: "Demo: Fallback Routing"**
    - Sends 8 requests/step with 20% completion rate
    - Watch the Decision Trace tab for FALLBACK_ACCEPTED entries
    - Event Log will show: "⚠️ FALLBACK ROUTING: chat request → o1 (primary gpt-4o-mini full)"

    ### 2️. Queue Management
    **Use Scenario: "Demo: Queue Buildup"**
    - Sends 15 requests/step with 10% completion rate
    - Watch Throttler State tab - Queued column will increase
    - Event Log will show: "🔴 QUEUED: req_X_Y to gpt-4o-mini (both primary and fallback full)"

    ### 3️. Queue Draining
    **Use Scenario: "Demo: Queue Draining"**
    - First build up queues using scenario 2
    - Then switch to "Demo: Queue Draining" scenario
    - Sends only 2 requests/step with 80% completion
    - Watch queues decrease as requests complete
    - Event Log will show: "✅ QUEUE DRAINED: req_X_Y promoted from queue to active on gpt-4o-mini"
    
    ## Architecture
    
    ### Model Throttler
    - Intent-aware routing (chat vs agent vs batch)
    - Primary + fallback model selection
    - Business-policy-driven decisions
    - Queue management and draining
    - Explainable routing decisions
    
    ### Gateway (Platform Layer)
    - Rate limiting (RPM/TPM)
    - Token enforcement
    - Retry logic and backoff
    - Network security
    - Connection pooling
    
    ## Routing Policy
    
    **Chat Requests:**
    - Primary: `gpt-4o-mini`
    - Fallback: `o1`
    
    **Agent Requests:**
    - Primary: `o1`
    - Fallback: `gpt-4o-mini`
    
    **Batch Requests:**
    - Primary: `gpt-4.1-mini`
    - Fallback: None (can be queued)
    
    ## Model Capacities
    
    - **gpt-4o-mini**: Max 3 concurrent
    - **o1**: Max 2 concurrent
    - **gpt-4.1-mini**: Max 1 concurrent
    
    ## Quick Test Procedure
    
    1. Click **Reset** to start fresh
    2. Select **"Demo: Fallback Routing"** scenario
    3. Click **Run Step** 3-5 times
    4. Go to **Event Log** tab - you'll see fallback routing
    5. Go to **Decision Trace** - filter by "FALLBACK_ACCEPTED"
    6. Select **"Demo: Queue Buildup"** scenario
    7. Click **Run Step** 3-5 times
    8. Go to **Throttler State** - see queued requests
    9. Select **"Demo: Queue Draining"** scenario
    10. Click **Run Step** - watch queues drain in real-time!
    
    ## What to Look For
    
    - **Fallback Routing**: Yellow rows in Decision Trace, ⚠️ warnings in Event Log
    - **Queue Buildup**: Red numbers in Throttler State "Queued" column
    - **Queue Draining**: ✅ success messages in Event Log showing promotions
    
    ## Technology Stack
    
    - **Python** for core logic
    - **Streamlit** for interactive UI
    - **Matplotlib** for visualizations
    - **Pandas** for data handling
    """)
