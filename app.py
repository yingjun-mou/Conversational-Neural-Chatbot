from tkinter import *
from chat import get_response, bot_name

# defined theme colors and fonts
BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"


class ChatApplication:

    def __init__(self):
        self.window = Tk()  # top-left widget
        self._setup_main_window()  # create layout

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=500, bg=BG_COLOR)

        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Welcome", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED) # freeze the widget

        # scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview) # if we scroll, the text displayed should move down

        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # message entry box
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT) # an entry widget to type in text
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus() # focus means when the app is opened, this entry box is by default selected
        self.msg_entry.bind("<Return>", self._on_enter_pressed) # <Return> corresponds to pressing return button event

        # send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    # define what we want it to behave when user press enter
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get() # grab the input text from entry box
        self._insert_messages(msg, "You")

    def _insert_messages(self, msg, sender):
        # if user accidentally hit enter without typing anything
        if not msg:
            return

        self.msg_entry.delete(0, END) # clear content in entry box and send it out

        # message from user
        msg1 = f"{sender}: {msg}\n\n" # if we don't add the new line, the next message will stay the same line
        self.text_widget.configure(state=NORMAL)  # unlock the widget before we modify it
        self.text_widget.insert(END, msg1) # insert the new message to the display widget
        self.text_widget.configure(state=DISABLED)  # lock it again

        # message from chatbot
        msg2 = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)  # unlock the widget before we modify it
        self.text_widget.insert(END, msg2)  # insert the new message to the display widget
        self.text_widget.configure(state=DISABLED)  # lock it again

        # scroll the widget to the end
        self.text_widget.see(END)


if __name__ == "__main__":
    app = ChatApplication()
    app.run()
