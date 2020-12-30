import time


def timer(function):
    """Time a function

    Args:
        function: function to time completion of.
    """

    def f(*args, **kwargs):
        before = time.time()
        function_return = function(*args, **kwargs)
        after = time.time()
        print(f"Time for function {function.__name__}: {round(after - before, 4)}s")
        return function_return

    return f
