import requests
import json
import numpy as np
"""
0	right
1	left

2	explosion
3	top
4	collapse
5	water
6	fire


7	person

8	red
9	blue
10	yellow
"""

def get_area_people_signal():
    area = 'red'
    people = 0
    signal = 'fire'
    url = 'http://127.0.0.1:5000/'
    r = requests.get(url)
    data = json.loads(r.text)
    if data[2] != 0:
        signal = 'explosion'
    if data[3] != 0:
        signal = 'top'
    if data[4] != 0:
        signal = 'collapse'
    if data[5] != 0:
        signal = 'water'
    if data[6] != 0:
        signal = 'fire'

    if data[7] != 0:
        people = data[7]

    if data[8] != 0:
        area = 'red'
    if data[9] != 0:
        area = 'blue'
    if data[10] != 0:
        area = 'yellow'
    return area,people,signal


def get_left_or_right():
    url = 'http://127.0.0.1:5000/'
    r = requests.get(url)
    data = json.loads(r.text)
    print(data)
    if data[0] == 1:
        return 'right'
    if data[1] == 1:
        return 'left'


if __name__ == '__main__':
    print(get_area_people_signal())
    # print(get_left_or_right())