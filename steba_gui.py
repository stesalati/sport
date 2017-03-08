#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Documentazione:
    http://tkinter.unpythonic.net/wiki/tkFileDialog
    http://zetcode.com/gui/tkinter/widgets/
"""


from Tkinter import Tk, Text, TOP, BOTH, X, N, LEFT, RIGHT, RAISED, BooleanVar, Checkbutton
from ttk import Frame, Label, Entry, Button, Style
import Tkconstants, tkFileDialog




class StebaGUI(Frame):
  
    def __init__(self, parent):
        Frame.__init__(self, parent)   
         
        self.parent = parent
        self.initUI(parent)

        
    def initUI(self, parent):
      
        
        
        self.parent.title("STEBA GUI")
        self.style = Style()
        self.style.theme_use("default")
        
        self.pack(fill=BOTH, expand=True)
        
        frame1 = Frame(self)
        frame1.pack(fill=X)
        
        lbl1 = Label(frame1, text="Title", width=6)
        lbl1.pack(side=LEFT, padx=5, pady=5)           
       
        entry1 = Entry(frame1)
        entry1.pack(fill=X, padx=5, expand=True)
        
        frame2 = Frame(self)
        frame2.pack(fill=X)
        
#        lbl2 = Label(frame2, text="Author", width=6)
#        lbl2.pack(side=LEFT, padx=5, pady=5)        
#
#        entry2 = Entry(frame2)
#        entry2.pack(fill=X, padx=5, expand=True)
#        
#        frame3 = Frame(self)
#        frame3.pack(fill=BOTH, expand=True)
#        
#        lbl3 = Label(frame3, text="Review", width=6)
#        lbl3.pack(side=LEFT, anchor=N, padx=5, pady=5)        
#
#        txt = Text(frame3)
#        txt.pack(fill=BOTH, pady=5, padx=5, expand=True)       


        self.var = BooleanVar()
        
        cb = Checkbutton(self, text="Show title",
            variable=self.var, command=self.onClick)
        cb.select()
        cb.place(x=50, y=50)
        
        Button(self, text='askopenfilename', command=self.askopenfilename).pack(fill=BOTH, padx=5, pady=5)


        
        
        
        closeButton = Button(self, text="Close")
        closeButton.pack(side=RIGHT, padx=5, pady=5)
        okButton = Button(self, text="OK")
        okButton.pack(side=RIGHT)
        
    def onClick(self):
        if self.var.get() == True:
            self.master.title("Checkbutton")
        else:
            self.master.title("")
            
    def askopenfilename(self):
        """Returns an opened file in read mode.
        This time the dialog just returns a filename and the file is opened by your own code.
        """
    
        # get filename
        filename = tkFileDialog.askopenfilename(defaultextension='.gpx',
                                                initialdir="",
                                                title='This is a title')
    
        # open file on your own
        if filename:
          return open(filename, 'r')
              

def main():
  
    root = Tk()
    root.geometry("300x300+300+300")
    app = StebaGUI(root)
    root.mainloop()  


if __name__ == '__main__':
    main()  