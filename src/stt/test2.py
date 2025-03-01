from beeply.notes import *


a = beeps()


a.hear('A_')


print("Done ")

a.hear("A")

def beeping(x):
    if x == 0:
        a.hear('A_')
    elif x ==1:
        a.hear("A")
    elif x ==2:
        a.hear('A_')
        a.hear('A_')
    elif x ==3:
        a.hear("A")
        a.hear("A")
    elif x ==4:
        a.hear('A_')
        a.hear("A")
    elif x ==5:
        a.hear("A")
        a.hear('A_')
    elif x ==6:
        a.hear('A_')
        a.hear("A")
        a.hear('A_')

for i in range(0,7):
    beeping(i)

