from collections.abc import Iterable
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
import time

def print_imgs(imgs, labels, cols=5, figsize=(7,7), title='', label_size=8):
    fig = plt.figure(figsize=figsize)
    rows = (len(imgs) + cols - 1) // cols  # calculate number of rows needed
    for i, (image, label) in enumerate(zip(imgs, labels)):
        ax = plt.subplot(rows,cols, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(label, fontdict={'fontsize': label_size})
        plt.axis("off")
    plt.suptitle(title)
    plt.show()
    return fig


class Timer:
    """
    count preformance time and print it.
    One can define a name while init a class exemplar to print that name along with the time.
    """
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        print(f"{self.name}:: {self.end - self.start:.6f} seconds")


def is_iterable(obj):
    return isinstance(obj, Iterable)


def list_or_none(obj):
    """"
    returns obj if obj is iterable, [] (an empty list) if obj is None, otherwise raises ValueError
    """
    if obj is None:
        return []
    elif is_iterable(obj):
        return obj
    else:
        raise ValueError('obj must be None or iterable')


def is_not_stop_word(token):
    return not (token.is_stop or
                token.is_punct or
                token.is_space or
                token.is_digit)

def is_subdict(subdict, dict):
    for key, val in subdict.items():
        try:
            if dict[key] != val:
                return False
        except KeyError:
            return False
    return True

def isnumber(number):
    try:
        float(number)
    except:
        return False
    else:
        return True


def generalized_mean(xs, p=1):
    if (not isinstance(p, int) and any(x < 0 for x in xs)):
        raise ValueError(
            f'cannot calculate generalized mean: the power ({p}) is non-integer and at least one of values is negative')

    if p < 0 and any(x == 0 for x in xs):
        raise ValueError(
            f'cannot calculate generalized mean: the power ({p}) is negative and and at least one of values is zero')
    if p == 0:
        return np.prod(xs) ** (1 / len(xs))
    else:
        return (np.sum(np.array(xs) ** p) / len(xs)) ** (1 / p)


def timeit(func):
    """
        Decorator that times the execution of a function.

        This decorator wraps a function and times how long it takes to execute.
        It prints the execution time in seconds to the console and returns the result of the function,
        along with the total time taken.

        Args:
            func (callable): The function to time.

        Returns:
            tuple: A tuple containing the result of the function and the time taken to execute it.

        Usage:
            @timeit
            def my_func():
                ...
        """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result, total_time
    return timeit_wrapper


class CountdownDecorator:
    """creates a countdown decorator"""

    def __init__(self, num):
        self.num = num
        self.counter = num
        self.end_string = '... '
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.counter == self.num:
                print(f'run {args[0]}')
            print(f'{self.counter}', end=self.end_string)
            result = func(*args, **kwargs)
            self.counter -= 1
            if self.counter <= 0:
                self.counter = self.num
                print('finished')
            return result
        return wrapper




