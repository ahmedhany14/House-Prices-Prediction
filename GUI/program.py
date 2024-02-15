from tkinter import *
import tkinter as tk
from tkinter import font as tkFont
from tkinter import ttk
from tkinter.ttk import *
from tkinter import *
from tkinter.ttk import *
from tkinter import ttk
from tkinter.messagebox import askyesno
from PIL import Image
import tkinter as tk
from tkinter import ttk as ttk
from tkinter import messagebox

# region DB
root = Tk()
w = 500
h = 300
padx = 10
pady = 10
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width - w) / 2)
y = int((screen_height - h) / 2)
root.geometry(f"{w}x{h}+{x}+{y}")
root.attributes("-fullscreen", True)
# endregion
# region Styles and Root
style = Style()
style.configure(
    "TButton",
    font=("calibri", 20, "bold"),
    borderwidth=8,
    activebackground="black",
    background="red",
)
# Changes will be reflected
# by the movement of mouse.
style.map(
    "TButton",
    foreground=[("active", "!disabled", "focus", "blue")],
    background=[("active", "Black")],
)
# *******************************************************
# *******************************************************
frame = Frame(root, width=screen_width, height=screen_height)
lbl_img = PhotoImage(
    file=r"GUI/photos/photo_5_2024-02-16_00-11-07.png"
)
lbl = Label(frame, image=lbl_img)
lbl.place(relheight=1, relwidth=1)
frame.place(relheight=1, relwidth=1)


# root.configure(bg="black")
# root.attributes("-fullscreen", True)# Set fullscreen
# Label1=Label(root,text="fgg").pack()
class PlaceholderEntry(ttk.Entry):
    """
    Custom modern Placeholder Entry box, takes positional argument master and placeholder along with\n
    textcolor(default being black) and placeholdercolor(default being grey).\n
    Use acquire() for getting output from entry widget\n
    Use shove() for inserting into entry widget\n
    Use remove() for deleting from entry widget\n
    Use length() for getting the length of text in the widget\n
    BUG 1: Possible bugs with binding to this class\n
    BUG 2: Anomalous behaviour with config or configure method
    """

    def __init__(
        self, master, placeholder, textcolor="black", placeholdercolor="grey", **kwargs
    ):
        self.text = placeholder
        self.__has_placeholder = False  # placeholder flag
        self.placeholdercolor = placeholdercolor
        self.textcolor = textcolor

        # style for ttk widget
        self.s = ttk.Style()

        # init entry box
        ttk.Entry.__init__(self, master, style="my.TEntry", **kwargs)
        self.s.configure("my.TEntry", forground=self.placeholdercolor)

        # add placeholder if box empty
        self._add()

        # bindings of the widget
        self.bind("<FocusIn>", self._clear)
        self.bind("<FocusOut>", self._add)
        self.bind_all("<Key>", self._normal)
        self.bind_all("<Button-1>", self._cursor)

    def _clear(self, *args):  # method to remove the placeholder
        if (
            self.get() == self.text and self.__has_placeholder
        ):  # remove placeholder when focus gain
            self.delete(0, tk.END)
            self.s.configure("my.TEntry", foreground="black", font=(0, 0, "normal"))
            self.__has_placeholder = False  # set flag to false

    def _add(self, *args):  # method to add placeholder
        if (
            self.get() == "" and not self.__has_placeholder
        ):  # if no text add placeholder
            self.s.configure(
                "my.TEntry", foreground=self.placeholdercolor, font=(0, 0, "bold")
            )
            self.insert(0, self.text)  # insert placeholder
            self.icursor(0)  # move insertion cursor to start of entrybox
            self.__has_placeholder = True  # set flag to true

    def _normal(self, *args):  # method to set the text to normal properties
        self._add()  # if empty add placeholder
        if (
            self.get() == self.text and self.__has_placeholder
        ):  # clear the placeholder if starts typing
            self.bind("<Key>", self._clear)
            self.icursor(-1)  # keep insertion cursor to the end
        else:
            self.s.configure(
                "my.TEntry", foreground=self.textcolor, font=(0, 0, "normal")
            )  # set normal font

    def acquire(self):
        """Custom method to get the text"""
        if self.get() == self.text and self.__has_placeholder:
            return "None"
        else:
            return self.get()

    def shove(self, index, string):
        """Custom method to insert text into entry"""
        self._clear()
        self.insert(index, string)

    def remove(self, first, last):
        """Custom method to remove text from entry"""
        if self.get() != self.text:
            self.delete(first, last)
            self._add()
        elif self.acquire() == self.text and not self.__has_placeholder:
            self.delete(first, last)
            self._add()

    def length(self):
        """Custom method to get the length of text in the entry widget"""
        if self.get() == self.text and self.__has_placeholder:
            return 0
        else:
            return len(self.get())

    def _cursor(
        self, *args
    ):  # method to not allow user to move cursor when placeholder exists
        if self.get() == self.text and self.__has_placeholder:
            self.icursor(0)


# endregion
# #########################################################################################################
House_area = Label(
    frame,
    text="House Area: ",
    width=23,
    foreground="#F4F6F7",
    background="#5C3317",
    font="Arial 20 bold",
    borderwidth=1,
)
Basement_Area = Label(
    frame,
    text="Basement Area:",
    width=23,
    foreground="#F4F6F7",
    background="#5C3317",
    font="Arial 20 bold",
)
Grade_living_area = Label(
    frame,
    text="Grade living area: ",
    width=23,
    foreground="#F4F6F7",
    background="#5C3317",
    font="Arial 20 bold",
    borderwidth=1,
)
Garage_Area = Label(
    frame,
    text="Garage Area: ",
    width=23,
    foreground="#F4F6F7",
    background="#5C3317",
    font="Arial 20 bold",
)
House_area_entry = PlaceholderEntry(
    frame, placeholder="Enter House Area", width=30, font="Arial 15 italic"
)
Basement_Area_entry = PlaceholderEntry(
    frame, placeholder="Enter Basement Area:", width=30, font="Arial 15 italic"
)
Grade_living_area_entry = PlaceholderEntry(
    frame, placeholder="Enter Grade living area: ", width=30, font="Arial 15 italic"
)
Garage_Area_entry = PlaceholderEntry(
    frame, placeholder="Enter Garage Area:", width=30, font="Arial 15 italic"
)
# ########################################################################################################
Garage_Capacite = Label(
    frame,
    text="garage capacite: ".title(),
    width=23,
    foreground="#F4F6F7",
    background="#5C3317",
    font="Arial 20 bold",
    borderwidth=1,
)
Number_Of_Rooms = Label(
    frame,
    text="Number Of Rooms:",
    width=23,
    foreground="#F4F6F7",
    background="#5C3317",
    font="Arial 20 bold",
)
Number_Of_Bathrooms = Label(
    frame,
    text="Number Of Bathrooms: ",
    width=23,
    foreground="#F4F6F7",
    background="#5C3317",
    font="Arial 20 bold",
    borderwidth=1,
)
Number_Of_Fireplaces = Label(
    frame,
    text="Number Of Fireplaces: ",
    width=23,
    foreground="#F4F6F7",
    background="#5C3317",
    font="Arial 20 bold",
)
Garage_Capacite_entry = PlaceholderEntry(
    frame, placeholder="Enter garage capacite:", width=30, font="Arial 15 italic"
)
Number_Of_Rooms_entry = PlaceholderEntry(
    frame, placeholder="Enter Number Of Rooms:", width=30, font="Arial 15 italic"
)
Number_Of_Bathrooms_entry = PlaceholderEntry(
    frame, placeholder="Enter Number Of Bathrooms:", width=30, font="Arial 15 italic"
)
Number_Of_Fireplaces_entry = PlaceholderEntry(
    frame, placeholder="Enter Number Of Fireplaces:", width=30, font="Arial 15 italic"
)
# ########################################################################################################
Year_Built = Label(
    frame,
    text="Year Built: ".title(),
    width=23,
    foreground="#F4F6F7",
    background="#5C3317",
    font="Arial 20 bold",
    borderwidth=1,
)
Year_Remodel_Add = Label(
    frame,
    text="Year Remodel Add:",
    width=23,
    foreground="#F4F6F7",
    background="#5C3317",
    font="Arial 20 bold",
)
Garage_Year_Build = Label(
    frame,
    text="Garage Year Build: ",
    width=23,
    foreground="#F4F6F7",
    background="#5C3317",
    font="Arial 20 bold",
    borderwidth=1,
)
Quality = Label(
    frame,
    text="Quality: ",
    width=23,
    foreground="#F4F6F7",
    background="#5C3317",
    font="Arial 20 bold",
)
Year_Built_entry = PlaceholderEntry(
    frame, placeholder="Enter Year Built:", width=30, font="Arial 15 italic"
)
Year_Remodel_Add_entry = PlaceholderEntry(
    frame, placeholder="Enter Year Remodel:", width=30, font="Arial 15 italic"
)
Garage_Year_Build_entry = PlaceholderEntry(
    frame, placeholder="Enter Garage Year Build:", width=30, font="Arial 15 italic"
)
Quality_entry = PlaceholderEntry(
    frame, placeholder="Number Quality:", width=30, font="Arial 15 italic"
)
# ########################################################################################################
# region Images and Functions
img = PhotoImage(file=r"GUI/photos/photo_6_2024-02-16_00-11-07.png")
img = img.subsample(10, 10)


def submit():
    # if(len(House_area_entry.get())>0 and len(Basement_Area_entry.get())>0 and len(Grade_living_area_entry.get())>0 and len(Garage_Area_entry.get())>0):
    #   cr.execute(f'insert into suppliers(sname, snumber, pname, pprice) values("{House_area_entry.get()}",{int(Basement_Area_entry.get())} , "{Grade_living_area_entry.get()}",{int(Garage_Area_entry.get())} )')
    #    db.commit()
    
    messagebox.showinfo("Added", "Added Succefully")


btn_add = Button(
    frame, text="Submit", compound="right", image=img, command=submit, width=12
)


# print(f"{screen_height} {screen_width}")
def exit():
    answer = askyesno(
        title="Exit Confirmation", message="Are you sure that you want to quit?".title()
    )
    if answer:
        root.destroy()


from tkinter import *

lbl = Label(
    frame,
    text="ML Project",
    width=30,
    bg="#5C3317",
    bd=5,
    relief="sunken",
    highlightbackground="red",
    font="Arial 40 bold",
    justify="center",
    fg="white",
    anchor="center",
).place(x=270, y=20)
from tkinter.ttk import *

# endregion
House_area.place(x=350, y=110)
Basement_Area.place(x=350, y=165)
Grade_living_area.place(x=350, y=220)
Garage_Area.place(x=350, y=275)
House_area_entry.place(x=800, y=110)
Basement_Area_entry.place(x=800, y=165)
Grade_living_area_entry.place(x=800, y=220)
Garage_Area_entry.place(x=800, y=275)
btn_add.place(x=750, y=785)
# ##################################################
Garage_Capacite.place(x=350, y=330)
Number_Of_Rooms.place(x=350, y=385)
Number_Of_Bathrooms.place(x=350, y=440)
Number_Of_Fireplaces.place(x=350, y=495)
Garage_Capacite_entry.place(x=800, y=330)
Number_Of_Rooms_entry.place(x=800, y=385)
Number_Of_Bathrooms_entry.place(x=800, y=440)
Number_Of_Fireplaces_entry.place(x=800, y=495)
# #################################################
Year_Built.place(x=350, y=550)
Year_Remodel_Add.place(x=350, y=605)
Garage_Year_Build.place(x=350, y=660)
Quality.place(x=350, y=715)
Year_Built_entry.place(x=800, y=550)
Year_Remodel_Add_entry.place(x=800, y=605)
Garage_Year_Build_entry.place(x=800, y=660)
Quality_entry.place(x=800, y=715)
# #################################################
# region The Rest
img5 = PhotoImage(file=r"GUI/photos/photo_2_2024-02-16_00-11-07.png")
img5 = img5.subsample(20, 20)


def clear():
    House_area_entry.delete(0, END)
    Basement_Area_entry.delete(0, END)
    Grade_living_area_entry.delete(0, END)
    Garage_Area_entry.delete(0, END)
    Garage_Capacite_entry.delete(0, END)
    Number_Of_Rooms_entry.delete(0, END)
    Number_Of_Bathrooms_entry.delete(0, END)
    Number_Of_Fireplaces_entry.delete(0, END)
    Year_Built_entry.delete(0, END)
    Year_Remodel_Add_entry.delete(0, END)
    Garage_Year_Build_entry.delete(0, END)
    Quality_entry.delete(0, END)


btn_home = Button(
    frame, text="Clear", compound="right", image=img5, command=clear
).place(x=500, y=800)
from tkinter import *

img2 = PhotoImage(file=r"GUI/photos/photo_1_2024-02-16_00-11-07.png")
img2 = img2.subsample(20, 20)
btn_exit = Button(
    frame,
    text="EXIT",
    compound="right",
    image=img2,
    width=90,
    command=exit,
    font="Arial 15 bold",
    bd=4,
    background="#EB5406",
    activebackground="Black",
    fg="white",
    padx=8,
)
btn_exit.place(x=15, y=15)
root.mainloop()
# endregion