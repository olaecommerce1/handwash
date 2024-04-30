import os
import cv2
import time
import warnings
import numpy as np

class ActionSteps:
    def __init__(self):
        self.placeholder_steps = {
            "STEP 1": "incomplete", "STEP 2 LEFT": "incomplete", "STEP 2 RIGHT": "incomplete",
            "STEP 3": "incomplete", "STEP 4 LEFT": "incomplete", "STEP 4 RIGHT": "incomplete",
            "STEP 5 LEFT": "incomplete", "STEP 5 RIGHT": "incomplete", "STEP 6 LEFT": "incomplete",
            "STEP 6 RIGHT": "incomplete", "STEP 7 LEFT": "incomplete", "STEP 7 RIGHT": "incomplete"
        }
        self.current_step_pointer = 0

    @property
    def steps(self):
        return self.placeholder_steps

    def is_correct_step(self, step_just_completed):
        next_steps = self.get_next_step().split(" or ")
        return step_just_completed in next_steps

    def get_step_number(self, current_step):
        return int(current_step.split(" ")[1])

    def incorrect_step_order(self, incorrect_step):
        next_step = self.get_next_step()
        if 'or' in next_step:
            message = "{0} is not the step, please perform either {1}."
        else:
            message = "{0} is not the step, please perform {1}."
        return message.format(incorrect_step, next_step)

    def all_steps_completed(self):
        return all(status == "complete" for status in self.steps.values())

    def action_completed_successfully(self):
        print("Actions completed successfully!")
        exit(0)

    def add_step(self, step_just_completed):
        if not self.is_correct_step(step_just_completed):
            print(self.incorrect_step_order(step_just_completed))
            return

        self.placeholder_steps[step_just_completed] = "complete"
        if self.all_steps_completed():
            self.action_completed_successfully()

        # Update the current step pointer
        self.current_step_pointer = next((i for i, status in enumerate(self.steps.values()) if status == "incomplete"), len(self.steps))

    def get_next_step(self):
        keys = list(self.steps.keys())
        current_key = keys[self.current_step_pointer]
        if "LEFT" in current_key:
            next_index = self.current_step_pointer + 1
            if next_index < len(keys) and "RIGHT" in keys[next_index]:
                right_key = keys[next_index]
                if self.steps[right_key] == "incomplete":
                    return f"{current_key} or {right_key}"
                return current_key
        return current_key

    def __del__(self):
        del self.placeholder_steps
        print("Steps system memory freed.")
