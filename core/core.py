from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Any
import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

# TODO: DIMENSIONS CAN HAVE OTHER SPACES
# TODO: IMPLEMENT EMPTY SPACE

# Dimension Class
class Dimension:
    def __init__(self, name: str, data_type: type = float, initial_value: Optional[float] = None):
        self.name = name
        self.data_type = data_type
        self.initial_value = initial_value

    def __repr__(self):
        return f"Dimension(name={self.name}, data_type={self.data_type.__name__})"

# Space Class
class Space:
    def __init__(self, name: str, dimensions: List[Dimension]):
        self.name = name
        self.dimensions = {dim.name: dim for dim in dimensions}

    def __repr__(self):
        return f"Space(name={self.name}, dimensions={list(self.dimensions.keys())})"

# Point Class
class Point:
    def __init__(self, space: Space, values: Optional[Dict[str, float]] = None):
        self.space = space
        self.values = {}
        # Initialize values
        for dim_name, dimension in self.space.dimensions.items():
            if values and dim_name in values:
                self.values[dim_name] = values[dim_name]
            elif dimension.initial_value is not None:
                self.values[dim_name] = dimension.initial_value
            else:
                self.values[dim_name] = dimension.data_type()
        # Validate data types
        for dim_name, value in self.values.items():
            expected_type = self.space.dimensions[dim_name].data_type
            if not isinstance(value, expected_type):
                raise TypeError(f"Value for dimension '{dim_name}' must be of type {expected_type.__name__}")

    def __repr__(self):
        return f"Point(space={self.space.name}, values={self.values})"

# Flow Class
class Flow:
    def __init__(self, points: Optional[Dict[str, Point]] = None):
        self.points = points if points else {}

    def add_point(self, name: str, point: Point):
        self.points[name] = point

    def get_point(self, name: str) -> Point:
        return self.points.get(name)

    def __repr__(self):
        return f"Flow(points={list(self.points.keys())})"

# Abstract Transform Class
class Transform(ABC):
    def __init__(self, name: str, domain: Space, codomain: Space):
        self.name = name
        self.domain = domain
        self.codomain = codomain

    @abstractmethod
    def apply(self, point_or_flow):
        pass

# LinearTransform Class
class LinearTransform(Transform):
    def __init__(self, name: str, domain: Space, codomain: Space, matrix: np.ndarray, bias: Optional[np.ndarray] = None):
        super().__init__(name, domain, codomain)
        self.matrix = matrix
        self.bias = bias if bias is not None else np.zeros((matrix.shape[0],))

    def apply(self, point: Point) -> Point:
        # Extract values from the point
        input_vector = np.array([point.values[dim] for dim in self.domain.dimensions])
        # Perform linear transformation
        output_vector = self.matrix @ input_vector + self.bias
        # Create output point
        output_values = {dim: val for dim, val in zip(self.codomain.dimensions, output_vector)}
        return Point(self.codomain, output_values)

# NonlinearTransform Class
class NonlinearTransform(Transform):
    def __init__(self, name: str, domain: Space, codomain: Space, function: Callable[[Dict[str, float]], Dict[str, float]]):
        super().__init__(name, domain, codomain)
        self.function = function

    def apply(self, point: Point) -> Point:
        # Apply the nonlinear function
        input_values = {dim: point.values[dim] for dim in self.domain.dimensions}
        output_values = self.function(input_values)
        # Validate output dimensions
        if set(output_values.keys()) != set(self.codomain.dimensions.keys()):
            raise ValueError("Output values do not match codomain dimensions.")
        return Point(self.codomain, output_values)

# ConstantTransform Class
class ConstantTransform(Transform):
    def __init__(self, name: str, codomain: Space, value: Dict[str, Any]):
        super().__init__(name, domain=None, codomain=codomain)
        self.value = value
    
    def apply(self, point_or_flow=None) -> Point:
        return Point(self.codomain, self.value)


# Block Class
class Block:
    def __init__(self, name: str, transform: Transform):
        self.name = name
        self.transform = transform
        self.inputs: List['Block'] = []
        self.outputs: List['Block'] = []

    def add_input(self, block: 'Block'):
        self.inputs.append(block)
        block.outputs.append(self)

    def process(self, flow: Flow) -> Flow:
        # Get the input point from the flow
        input_point = flow.get_point(self.transform.domain.name)
        if not input_point:
            raise ValueError(f"Input point '{self.transform.domain.name}' not found in flow.")
        # Apply the transform
        output_point = self.transform.apply(input_point)
        # Create new flow with the output point
        new_flow = Flow(flow.points.copy())
        new_flow.add_point(self.transform.codomain.name, output_point)
        return new_flow

    def __repr__(self):
        return f"Block(name={self.name}, transform={self.transform.name})"
    
# CompositeBlock Class
class CompositeBlock(Block):
    def __init__(self, name: str, internal_blocks: List[Block], connections: List[tuple], input_spaces: List[Space], output_spaces: List[Space]):
        super().__init__(name=name, transform=None)
        self.input_spaces = input_spaces
        self.output_spaces = output_spaces
        self.internal_pipeline = Pipeline()
        for block in internal_blocks:
            self.internal_pipeline.add_block(block)
        # Add connections between internal blocks
        for from_block, to_block in connections:
            self.internal_pipeline.connect_blocks(from_block, to_block)

    def process(self, flow: Flow) -> Flow:
        # Prepare the internal flow with inputs from the main flow
        internal_flow = Flow()
        for space in self.input_spaces:
            point = flow.get_point(space.name)
            if point:
                internal_flow.add_point(space.name, point)
            else:
                raise ValueError(f"Input point '{space.name}' not found in flow.")
        # Run the internal pipeline
        internal_output_flow = self.internal_pipeline.run(internal_flow)
        # Extract the output points and add them to the main flow
        for space in self.output_spaces:
            output_point = internal_output_flow.get_point(space.name)
            if output_point:
                flow.add_point(space.name, output_point)
            else:
                raise ValueError(f"Output point '{space.name}' not found after processing composite block.")
        return flow


# SplitterBlock Class
class SplitterBlock(Block):
    def __init__(self, name: str, input_space: Space, output_spaces: List[Space]):
        super().__init__(name=name, transform=None)
        self.input_space = input_space
        self.output_spaces = output_spaces
    
    def process(self, flow: Flow) -> Flow:
        input_point = flow.get_point(self.input_space.name)
        if not input_point:
            raise ValueError(f"Input point '{self.input_space.name}' not found in flow.")
        # Duplicate the input point to all output spaces
        for space in self.output_spaces:
            duplicated_point = Point(space, input_point.values.copy())
            flow.add_point(space.name, duplicated_point)
        return flow

# CombinerBlock Class
class CombinerBlock(Block):
    def __init__(self, name: str, input_spaces: List[Space], output_space: Space):
        super().__init__(name=name, transform=None)
        self.input_spaces = input_spaces
        self.output_space = output_space
    
    def process(self, flow: Flow) -> Flow:
        combined_values = {}
        for space in self.input_spaces:
            point = flow.get_point(space.name)
            if point:
                combined_values.update(point.values)
            else:
                raise ValueError(f"Point '{space.name}' not found in flow.")
        combined_point = Point(self.output_space, combined_values)
        flow.add_point(self.output_space.name, combined_point)
        return flow

# OutputTypeBlock Class
class OutputTypeBlock(Block):
    def __init__(self, name: str, input_space: Space, output_space: Space):
        super().__init__(name=name, transform=None)
        self.input_space = input_space
        self.output_space = output_space
    
    def process(self, flow: Flow) -> Flow:
        input_point = flow.get_point(self.input_space.name)
        if not input_point:
            raise ValueError(f"Input point '{self.input_space.name}' not found in flow.")
        # For this example, we assume it's an identity transform
        output_point = Point(self.output_space, input_point.values.copy())
        flow.add_point(self.output_space.name, output_point)
        return flow

# Pipeline Class
class Pipeline:
    def __init__(self):
        self.blocks: Dict[str, Block] = {}
        self.graph = nx.DiGraph()

    def add_block(self, block: Block):
        self.blocks[block.name] = block
        self.graph.add_node(block.name)

    def connect_blocks(self, from_block: Block, to_block: Block):
        from_block.outputs.append(to_block)
        to_block.inputs.append(from_block)
        self.graph.add_edge(from_block.name, to_block.name)

    def _topological_sort(self) -> List[str]:
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("The pipeline graph has cycles and cannot be sorted topologically.")

    def run(self, initial_flow: Flow) -> Flow:
        execution_order = self._topological_sort()
        flow = initial_flow
        for block_name in execution_order:
            block = self.blocks[block_name]
            flow = block.process(flow)
        return flow

    def run_parallel(self, initial_flow: Flow) -> Flow:
        execution_order = self._topological_sort()
        flow = initial_flow.copy()
        with ThreadPoolExecutor() as executor:
            futures = {}
            for block_name in execution_order:
                block = self.blocks[block_name]
                futures[block_name] = executor.submit(block.process, flow)
            # Collect results
            for block_name in execution_order:
                flow = futures[block_name].result()
        return flow

    def __repr__(self):
        return f"Pipeline(blocks={list(self.blocks.keys())})"
