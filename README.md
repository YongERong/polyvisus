# PolyVisus⌨️
Revolutionizing Typing, One Infuriating Tap at a Time.

## Inspiration
In a world dominated by convenience and efficiency, we asked ourselves one bold, revolutionary question: "What if we could make typing infinitely worse?" While the rest of humanity tirelessly innovates to improve productivity, we felt that the modern typing experience lacked the chaos and frustration of life's true unpredictability. 

Inspired by the randomness of spilled soup on a keyboard and the tactile delight of slapping a table in rage, PolyVisus was born.

## What it does
PolyVisus uses Computer Vision to allow you type on any flat surface—but with a twist. The keyboard layout isn’t static; it shuffles unpredictably after every few keystrokes. Gone are the days of mundane QWERTY dominance! Now, every typing session is a journey into chaos, where typing "hello" might require finding *H* in the bottom-left corner and *E* halfway up the table.

## Challenges we ran into
1. Users Refused to Use It: We underestimated how few people would willingly participate in our typing torment. Turns out, most people enjoy efficiency. Who knew?
2. Frustration-Induced Hardware Damage: Testers frequently slammed their tables in anger, confusing the model into thinking they were typing "ASDFASDF."
3. Latency-Induced Chaos: The deliberate delay in key recognition was so annoying that even we, the creators, struggled to demo it without swearing.

## Accomplishments that we're proud of
We achieved our primary goal of making typing as painful and counterproductive as humanly possible.

## What we learned
Building a project no one needs or wants is oddly fulfilling, especially when it works exactly as horribly as intended.

## Getting Started
After reading this, if you still decide to continue down this path of pain:
```
git clone https://github.com/YongERong/polyvisus.git
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd polyvisus/server
python keyboard.py
```
