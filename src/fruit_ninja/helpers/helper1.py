import tkinter as tk

def create_reference_window(width=640, height=384, x=100, y=100):
    # Ensure width and height are multiples of 32
    if width % 32 != 0 or height % 32 != 0:
        raise ValueError("Both width and height must be multiples of 32")

    root = tk.Tk()
    root.title(f"{width}x{height} Reference Frame")

    # Place window at given coordinates on screen
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.resizable(False, False)

    # Create canvas
    canvas = tk.Canvas(root, width=width, height=height, bg="white")
    canvas.pack()

    # Draw rectangle outline
    canvas.create_rectangle(
        2, 2, width-2, height-2,
        outline="black",
        width=2
    )

    root.mainloop()

if __name__ == "__main__":
    # Example: 640x384 window placed 100px right and 100px down
    create_reference_window(640, 384, 100, 100)
