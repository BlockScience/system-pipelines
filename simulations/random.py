# %%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from core import (Dimension, Space, Point, Flow, Block, StatefulTransform, SignalGeneratorBlock, Pipeline)

# %%
# Define the Position Space for 1D Random Walk
position_dimension = Dimension('x')
position_space = Space('PositionSpace1D', [position_dimension])

# Initial Point
initial_position = Point(position_space, {'x': 0.0})

# Random Step Signal Generator Block
import random

def random_step_function(state):
    step = random.choice([-1, 1])  # Random step of -1 or +1
    output_values = {'step': step}
    return output_values, state

step_dimension = Dimension('step')
step_space = Space('StepSpace', [step_dimension])

random_step_block = SignalGeneratorBlock(
    name='RandomStepGenerator',
    codomain=step_space,
    signal_function=random_step_function
)


# %%
# Position Update Block using StatefulTransform
class RandomWalkBlock(Block):
    def __init__(self, name: str, position_space: Space, step_space: Space):
        def update_position_function(input_values, state):
            # input_values contains 'step' from the step_space
            step = input_values['step']
            position = state.get('position', 0.0)
            new_position = position + step
            state['position'] = new_position
            output_values = {'x': new_position}
            return output_values, state

        transform = StatefulTransform(
            name=f"{name}_transform",
            domain=step_space,
            codomain=position_space,
            function=update_position_function,
            initial_state={'position': initial_position.values['x']}
        )
        super().__init__(name, transform)

random_walk_block = RandomWalkBlock(
    name='RandomWalk',
    position_space=position_space,
    step_space=step_space
)


# %%
# Build the Pipeline
pipeline = Pipeline()
pipeline.add_block(random_step_block)
pipeline.add_block(random_walk_block)
pipeline.connect_blocks(random_step_block, random_walk_block)

# Initial Flow
initial_flow = Flow()

# Run the Random Walk for 10 steps
flow = initial_flow
for _ in range(10):
    flow = pipeline.run(flow)
    current_position = flow.get_point('PositionSpace1D').values['x']
    print(f"Current Position: {current_position}")



