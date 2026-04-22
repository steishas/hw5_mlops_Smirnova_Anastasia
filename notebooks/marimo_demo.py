import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    x = 10
    return (x,)


@app.cell
def _(x):
    y = x * 2
    return (y,)


@app.cell
def _(y):
    y
    return


if __name__ == "__main__":
    app.run()
