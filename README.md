# Control System Pipeline Modeling

## Core Library Overview

The core library implements a flexible and modular **control system pipeline**. It is designed to handle various types of signal transformations and flows, making it suitable for control system tasks where data needs to be transformed and managed efficiently.

### Key Components

1. **Dimension**:

   - Represents a single dimension (or variable) in the control system.
   - Properties: name, data type, and optional initial value.

2. **Space**:

   - A collection of dimensions that define the state space of the control system.

3. **Point**:

   - Represents a specific point (or state) in a particular space with assigned values for each dimension.

4. **Flow**:

   - Manages multiple points, enabling the propagation of signals through the control system.

5. **Transform** (Abstract base class):

   - **LinearTransform**: Applies linear transformations to the system's state or input signals, often used in linear control systems or state-space representations.
   - **NonlinearTransform**: Applies non-linear transformations, typical in systems with nonlinear dynamics or signal processing.
   - **ConstantTransform**: Outputs a constant signal, useful for reference signals or fixed setpoints in a control loop.

6. **Block** (Base class for processing units):

   - **CompositeBlock**: Combines multiple processing units or control blocks into a single functional block, representing a complex system or subsystem.
   - **SplitterBlock**: Duplicates an input signal to distribute it across multiple subsystems, useful in feedback or feedforward control architectures.
   - **CombinerBlock**: Merges multiple input signals into a single output, such as when combining feedback signals.
   - **OutputTypeBlock**: Converts signal types, ensuring that outputs conform to the desired format, like converting between continuous and discrete signals.

7. **Pipeline**:
   - Manages the overall flow of control blocks and ensures that signals are processed sequentially or in parallel, as required by the control system design.

### Key Features

- **Modular design**: Allows easy composition of complex control flows and subsystems.
- **Linear and non-linear transformations**: Support for a wide range of system dynamics.
- **Flexible state spaces**: Handles multiple dimensions and variables efficiently.
- **Type-safe signal handling**: Ensures correct data types are maintained across transformations.
- **Parallel execution**: Capable of processing multiple control blocks simultaneously using parallel threads for real-time control applications.

## Simulation: Control System Use Case

The `01_control_system.ipynb` notebook demonstrates a practical application of the core library in a control system setting. Here's an overview of the simulation:

### Requirements and Installation

This project requires Python 3.6 or later. To install the necessary dependencies, follow these steps:

1. Ensure you have Python 3.6+ installed on your system.

2. Clone this repository:

   ```
   git clone https://github.com/BlockScience/system-pipelines.git
   cd system-pipelines
   ```

3. (Optional) Create and activate a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

The `requirements.txt` file includes the following dependencies:

- numpy
- networkx

Note: This project uses type hints, which is why Python 3.6 or later is required.

1. **Setup**:

   - Define input and output spaces representing system states or control signals.
   - Create transformation functions (f1, f2, f3) to model system behavior or control actions.

2. **Block Creation**:

   - Implement individual blocks for each transformation, representing different parts of the control system (e.g., plant dynamics, controllers, sensors).
   - Create a combiner block to merge signals, simulating how multiple feedback signals or control inputs interact.

3. **Composite Block**:

   - Assemble a `CompositeBlock` named 'SystemBlock', encapsulating multiple blocks into a single control system module, which could represent a full control loop or subsystem.

4. **Pipeline Execution**:

   - Set up the main control pipeline with the composite block.
   - Provide input signals as a `Point` object, simulating sensor data or control inputs.
   - Run the pipeline to process the inputs and retrieve the system's output, simulating the system's response to the inputs.

5. **Results**:
   - The system transforms the input signals `{'u1': 5.0, 'u2': True}` into the output `{'output_str': 'Result: 5.0', 'output_float': 10.0, 'output_bool': True}`, demonstrating how control signals flow through the pipeline and are processed by the blocks.

This simulation highlights how the core library can be used to model complex control systems with multiple transformations, signal flows, and interactions between various system components. The system can be encapsulated within a composite block and executed efficiently through the control pipeline.
