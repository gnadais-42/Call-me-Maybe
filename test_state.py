from src.state_machine import StateMachine


machine = StateMachine()

print(machine.context.state)

machine.transition()
print(machine.context.state)

machine.transition()
print(machine.context.state)