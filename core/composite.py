from typing import Callable, Dict, List, Optional, Any
from .core import Transform, Block, Point, Space, Dimension, Flow, Pipeline, CompositeBlock
from .core import EMPTY_SPACE
# StatefulTransform Class
class StatefulTransform(Transform):
    def __init__(self, name: str, domain: Space, codomain: Space, function: Callable, initial_state: Optional[Dict[str, Any]] = None):
        super().__init__(name, domain, codomain)
        self.function = function
        self.state = initial_state if initial_state else {}
    
    def apply(self, point: Optional[Point]) -> Point:
        if self.domain is None or point is None:
            input_values = {}
        else:
            input_values = {dim: point.values[dim] for dim in self.domain.dimensions}
        result = self.function(input_values, self.state)
        
        # Check if the result is a tuple (output_values, new_state)
        if isinstance(result, tuple) and len(result) == 2:
            output_values, self.state = result
        else:
            output_values = result

        # Validate output dimensions
        if set(output_values.keys()) != set(self.codomain.dimensions.keys()):
            raise ValueError("Output values do not match codomain dimensions.")
        return Point(self.codomain, output_values)


# MemoryBlock Class
class MemoryBlock(Block):
    def __init__(self, name: str, space: Space, initial_memory: Optional[Dict[str, Any]] = None):
        # Define a StatefulTransform that acts as memory
        def memory_function(input_values, state):
            output_values = state.copy()
            state.update(input_values)
            return output_values, state
        
        transform = StatefulTransform(
            name=f"{name}_transform",
            domain=space,
            codomain=space,
            function=memory_function,
            initial_state=initial_memory if initial_memory else {dim: 0 for dim in space.dimensions}
        )
        super().__init__(name, transform)

# SamplerBlock Class
class SamplerBlock(Block):
    def __init__(self, name: str, space: Space, sampling_interval: float):
        # Initialize time and last sample time
        self.current_time = 0.0
        self.last_sample_time = -sampling_interval  # Ensures sampling at time zero
        self.sampling_interval = sampling_interval
        
        # Define a stateful function
        def sampler_function(input_values, state):
            self.current_time += self.sampling_interval
            if self.current_time - self.last_sample_time >= self.sampling_interval:
                self.last_sample_time = self.current_time
                output_values = input_values
            else:
                output_values = {dim: None for dim in space.dimensions}
            return output_values, state  # State is unused here
        
        transform = StatefulTransform(
            name=f"{name}_transform",
            domain=space,
            codomain=space,
            function=sampler_function
        )
        super().__init__(name, transform)

# Example: IntegratorBlock as a Stateful Block
class IntegratorBlock(Block):
    def __init__(self, name: str, space: Space):
        def integrator_function(input_values, state):
            # Simple integrator: state += input * dt
            dt = 1.0  # Assume a fixed time step for simplicity
            state['integral'] += input_values['value'] * dt
            output_values = {'integral': state['integral']}
            return output_values, state
        
        initial_state = {'integral': 0.0}
        codomain = Space(name=f"{space.name}_integral", dimensions=[Dimension('integral')])
        transform = StatefulTransform(
            name=f"{name}_transform",
            domain=space,
            codomain=codomain,
            function=integrator_function,
            initial_state=initial_state
        )
        super().__init__(name, transform)

# Modified Pipeline to Handle Feedback Loops
class FeedbackPipeline(Pipeline):
    def _topological_sort(self) -> List[str]:
        # Allow cycles in the graph
        return list(self.blocks.keys())  # Simple execution order
    
    def run(self, initial_flow: Flow, iterations: int = 1) -> Flow:
        flow = initial_flow
        for _ in range(iterations):
            for block_name in self.blocks:
                block = self.blocks[block_name]
                flow = block.process(flow)
        return flow

# FeedbackLoopBlock Class
class FeedbackLoopBlock(CompositeBlock):
    def __init__(self, name: str, internal_blocks: List[Block], connections: List[tuple], input_spaces: List[Space], output_spaces: List[Space]):
        super().__init__(name, internal_blocks, connections, input_spaces, output_spaces)
        # Use FeedbackPipeline instead of Pipeline
        self.internal_pipeline = FeedbackPipeline()
        for block in internal_blocks:
            self.internal_pipeline.add_block(block)
        for from_block, to_block in connections:
            self.internal_pipeline.connect_blocks(from_block, to_block)
    
    def process(self, flow: Flow) -> Flow:
        # Similar to CompositeBlock, but specify iterations for feedback
        internal_flow = Flow()
        for space in self.input_spaces:
            point = flow.get_point(space.name)
            if point:
                internal_flow.add_point(space.name, point)
            else:
                raise ValueError(f"Input point '{space.name}' not found in flow.")
        internal_output_flow = self.internal_pipeline.run(internal_flow, iterations=10)
        for space in self.output_spaces:
            output_point = internal_output_flow.get_point(space.name)
            if output_point:
                flow.add_point(space.name, output_point)
            else:
                raise ValueError(f"Output point '{space.name}' not found after processing feedback loop.")
        return flow

# EventDrivenBlock Class
class EventDrivenBlock(Block):
    def __init__(self, name: str, domain: Space, codomain: Space, event_condition: Callable, event_action: Callable):
        def event_function(input_values, state):
            if event_condition(input_values):
                output_values = event_action(input_values)
            else:
                output_values = {dim: None for dim in codomain.dimensions}
            return output_values, state  # State is unused here
        
        transform = StatefulTransform(
            name=f"{name}_transform",
            domain=domain,
            codomain=codomain,
            function=event_function
        )
        super().__init__(name, transform)

# SignalGeneratorBlock Class
# SignalGeneratorBlock Class
class SignalGeneratorBlock(Block):
    def __init__(self, name: str, codomain: Space, signal_function: Callable, initial_state: Optional[Dict[str, Any]] = None):
        self.domain = EMPTY_SPACE
        def generator_function(input_values, state):
            # input_values is unused; this block generates its own output
            output_values = signal_function(state)
            state['t'] += state.get('dt', 1.0)
            return output_values, state
        
        default_state = {'t': 0.0, 'dt': 1.0}
        if initial_state:
            default_state.update(initial_state)

        transform = StatefulTransform(
            name=f"{name}_transform",
            domain=self.domain,  # Use EMPTY_SPACE instead of creating a new empty space
            codomain=codomain,
            function=generator_function,
            initial_state=default_state
        )
        super().__init__(name, transform)
    
    def process(self, flow: Flow) -> Flow:
        # Since there is no input, we directly apply the transform
        output_point = self.transform.apply(None)
        flow.add_point(self.transform.codomain.name, output_point)
        return flow


# DelayBlock Class
class DelayBlock(Block):
    def __init__(self, name: str, space: Space, delay_steps: int):
        def delay_function(input_values, state):
            buffer = state.get('buffer', [])
            buffer.append(input_values)
            if len(buffer) > delay_steps:
                output_values = buffer.pop(0)
            else:
                output_values = {dim: 0 for dim in space.dimensions}
            state['buffer'] = buffer
            return output_values, state
        
        initial_state = {'buffer': []}
        transform = StatefulTransform(
            name=f"{name}_transform",
            domain=space,
            codomain=space,
            function=delay_function,
            initial_state=initial_state
        )
        super().__init__(name, transform)
