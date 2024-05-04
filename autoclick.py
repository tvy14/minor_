import time
import pyautogui
import random

while True:
    random_time=random.uniform(0.5, 3.0)
    time.sleep(random_time)
    pyautogui.click()