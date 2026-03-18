import tkinter as tk

def create_reference_window():
    # Create the main window
    root = tk.Tk()
    root.title("640x640 Reference Window")

    # Set fixed size
    root.geometry("640x640")
    root.resizable(False, False)  # Prevent resizing

    # Create a white canvas filling the window
    canvas = tk.Canvas(root, width=640, height=640, bg="white")
    canvas.pack()

    # Run the window loop
    root.mainloop()

if __name__ == "__main__":
    create_reference_window()
