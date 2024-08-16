# Cascade models use a "decider" function to determine whether the output is valid or not.
# If the output is valid, we return it. Otherwise, we move to the next step.
# For chat completion models, we allow the decider to return a "correction", and let the prompt have another go.
from dataclasses import dataclass
from typing import Any

class CascadeStep:
    CORRECTABLE = False

    def __init__(self):
        pass

    def original(x):
        raise NotImplementedError()
    
    def corrected(x):
        raise NotImplementedError()
    
class Decider:
    def __init__(self):
        pass

    def __call__(self, x):
        raise NotImplementedError()
    
@dataclass
class CascadeStepResult:
    original_text: str
    step_name: str
    step_iteration: int
    step_iteration_description: str
    result: Any
    decision: str
    correction: str


class Cascade:
    def __init__(self, steps, decider):
        self.steps = steps
        self.decider = decider

    def __call__(self, x):
        results = []
        done = False
        for step in self.steps:
            for i, (result, (decision, correction, iter_description)) in enumerate(self.run_step(step, x)):
                results.append(CascadeStepResult(
                    original_text=x,
                    step_name=step.__class__.__name__,
                    step_iteration=i,
                    step_iteration_description=iter_description,
                    result=result,
                    decision=decision,
                    correction=correction
                ))
                if decision == "Good":
                    done = True
                    break
            if done:
                break

        # New API
        # Return the first "good" result, if there is one
        for result in results:
            if result.decision == "Good":
                return result, results
            
        # Return the first "okay" result, if there is one
        for result in results:
            if result.decision == "Okay":
                return result, results
            
        # We failed
        return None, results
    
    def run_step(self, step, x):
        # Try the original version of the step
        # print("*** Running step", step.__class__.__name__, "***")
        # print("Input:", x)
        result = step.original(x)
        # print("First Attempt:", result)
        decision, correction = self.decider(x, result)
        # print("  Decision:", decision)
        # print("  Correction:", correction)
        # print()
        yield result, (decision, correction, "original")

        # Return None if the step is not correctable
        if not step.CORRECTABLE:
            return None, (None, None, None)
        
        # Try the corrected version of the step
        prev = result
        result = step.corrected(x, prev, correction)
        # print("Corrected Attempt:", result)
        decision, correction = self.decider(x, result)
        # print("  Decision:", decision)
        # print("  Correction:", correction)
        # print()
        yield result, (decision, correction, "corrected")
        
        # Step has failed
        return None, (None, None, None)
