import itertools


colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
linestyles = ['-', '--', '-.', ':']

# 自动生成 styles
style_cycle = itertools.cycle(itertools.product(linestyles, colors))


def build_styles():
    return [(color, linestyle) for linestyle in linestyles for color in colors]