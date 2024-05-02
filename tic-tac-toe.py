import os.path
from tkinter import *
import sys
import time
from collections import namedtuple

#sys.path.append('../')
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from games import *

gBoard = None
root = None
buttons = []
frames = []
x_pos = []
o_pos = []
count = 0
sym = ""
result = None
choices = None
gheight = 4
gwidth = 4
gkmatch = 4
gdepth = -1 # -1 means search cutoff depth set to leaf level.



def create_frames(root):
    """
    This function creates the necessary structure of the game.
    """
    global gBoard
    gBoard = TicTacToe(gwidth, gheight, gkmatch, gdepth)
   
    for _ in range(gheight):
        framei = Frame(root)
        create_buttons(framei)
        framei.pack(side = BOTTOM)
        frames.append(framei)

    uiFrame = Frame(root)
    buttonExit = Button(uiFrame, height=1, width=4, text="Exit", command=lambda: exit_game(root))
    buttonExit.pack(side=LEFT)
    uiFrame.pack(side=TOP)
    buttonReset = Button(uiFrame, height=1, width=4, text="Reset", command=lambda: reset_game())
    buttonReset.pack(side=LEFT)

    """Depth is used with AlphaBeta and Expectimax players"""
    depthFrame = Frame(root) 
    depthFrame.pack(side=TOP)
    depthLabel = Label(depthFrame, text="Depth:", width=10, fg="green")
    depthLabel.pack(side=LEFT)

    depthStr = StringVar()
    depthStr.set(str("-1"))  # -1 means search depth set to max, meaning to terminal state.
    
    def depthCallback(event):
        global gdepth
        dstr = event.widget.get().strip()
        print("depth is ", dstr)
        if dstr.isdigit():
            gdepth = int(dstr)
            print("gdepth is ", gdepth)

        return True
    
    depthEntry = Entry(depthFrame, width =7, textvariable=depthStr)
    depthEntry.bind('<KeyRelease>', depthCallback)
    depthEntry.pack(side = LEFT)
    


    



butCount = 0
def create_buttons(frame):
    """
    creates the buttons for one row of the board .
    """
    buttons_in_frame = []
    global butCount

    for _ in range(gwidth):
        button = Button(frame, bg = "yellow", height=1, width=2, text=" ", padx=2, pady=2)
        button.config(command=lambda btn=button: on_click(btn))
        button.pack(side=LEFT)
        buttons_in_frame.append(button)
        butCount += 1

    buttons.append(buttons_in_frame)


def on_click(button):
    """
    This function determines the action of any button.
    """
    global gBoard, choices, count, sym, result, x_pos, o_pos
    print("onClick: button.text=", button['text'])
    if count % 2 == 0:
        sym = "X"
    else:
        sym = "O"
    count += 1

    print("depth set at ", gdepth)
    gBoard.d = gdepth
    button.config(text=sym, state='disabled', disabledforeground="red")  # For cross

    x, y = get_coordinates(button)
    x += 1
    y += 1
    x_pos.append((x, y))
    state1 = gen_state(to_move=sym, x_positions=x_pos,  o_positions=o_pos, h=gBoard.k, v=gBoard.k)

    #check human player victory:
    if gBoard.compute_utility(state1.board.copy(), (x, y), state1.to_move):
        result.set("You win :)")
        disable_game()
        return

    result.set("O Turn!")
    Tk.update(root)
    time.sleep(0.7)
 
    if count % 2 == 0:  # Used again, will become handy when user is given the choice of turn.
        sym = "X"
    else:
        sym = "O"
    count += 1
    state2 = gen_state(to_move=sym, x_positions=x_pos, o_positions=o_pos, h=gBoard.k, v=gBoard.k)

    try:
        choice = choices.get()
        if "Random" in choice:
            a, b = random_player(gBoard, state2)
        elif "Expectimax" in choice:
            a, b = expect_minmax(gBoard, state2)
        elif "MinMax" in choice:
            a, b = minmax_player(gBoard, state2)
        else:
            a, b = alpha_beta_player(gBoard, state2)
    except (ValueError, IndexError, TypeError) as e:
        disable_game()
        result.set("It's a draw :|")
        return
    if 1 <= a <= gwidth and 1 <= b <= gheight:
        o_pos.append((a, b))
        button_to_change = get_button(a - 1, b - 1)

        board = state2.board.copy()
        move = (a, b)
        board[move] = sym
        button_to_change.config(text=sym, state='disabled', disabledforeground="black")

        if gBoard.compute_utility(board, move, sym) == -gBoard.k:
            result.set("You lose :(")
            disable_game()
        else:
            result.set("Your Turn!")



# Not Used anymore: Replaced by "k_in_row" function.
def check_victory(button):
    """
    This function checks various winning conditions of the game.
    """
    # check if previous move caused a win on vertical line
    x, y = get_coordinates(button)
    tt = button['text']
    verticalWin = True
    for i in range(gkmatch):
        if buttons[0][y]['text'] != buttons[i][y]['text']:
            verticalWin = False
            break

    if verticalWin == True:
        for i in range(gkmatch):
            buttons[i][y].config(text="|" + tt + "|")
        return True

    # check if previous move caused a win on horizontal line
    horizontalWin = True
    for i in range(gkmatch):
        if buttons[x][0]['text'] != buttons[x][i]['text']:
            horizontalWin = False
            break

    if horizontalWin == True:
        for i in range(gkmatch):
            buttons[x][i].config(text="--" + tt + "--")
        return True


    # check if previous move was on the main diagonal and caused a win
    if x == y:
        diagonalWin = True
        for i in range(gkmatch):
            if buttons[0][0]['text'] != buttons[i][i]['text']:
                diagonalWin = False
                break

        if diagonalWin == True:
            for i in range(gkmatch):
                buttons[i][i].config(text="\\" + tt + "\\")
            return True


    # check if previous move was on the secondary diagonal and caused a win
    maxIndx = gkmatch - 1
    if x + y == maxIndx:
        diagonalWin = True
        for i in range(gkmatch):
            if buttons[0][maxIndx]['text'] != buttons[i][maxIndx-i]['text']:
                diagonalWin = False
                break

        if diagonalWin == True:
            for i in range(gkmatch):
                buttons[i][maxIndx-i].config(text="/" + tt + "/")
            return True


    return False


def get_coordinates(button):
    """
    This function returns the coordinates of the button clicked.
    """
    for x in range(len(buttons)):
        for y in range(len(buttons[x])):
            if buttons[x][y] == button:
                return x, y


def get_button(x, y):
    """
    This function returns the button memory location corresponding to a coordinate.
    """
    return buttons[x][y]


def reset_game():
    """
    This function will reset all the tiles to the initial null value.
    """
    global x_pos, o_pos, frames, count

    count = 0
    x_pos = []
    o_pos = []
    result.set("Your Turn!")
    for x in frames:
        for y in x.winfo_children():
            y.config(text=" ", state='normal')


def disable_game():
    """
    This function deactivates the game after a win, loss or draw.
    """
    global frames
    for x in frames:
        for y in x.winfo_children():
            y.config(state='disabled')


def exit_game(root):
    """
    This function will exit the game by killing the root.
    """
    root.destroy()


if __name__ == "__main__":
    if len(sys.argv) == 4:
        gheight = int(sys.argv[1])
        gwidth =  int(sys.argv[2])
        gkmatch = int(sys.argv[3])
    else:
        gheight = gwidth = gkmatch = 3

    root = Tk()
    root.title("TicTacToe")
    width = gwidth * 80
    height = gheight * 80
    geoStr = str(width) + "x" + str(height)
    root.geometry(geoStr)  
    root.resizable(0, 0)  # To remove the maximize window option
    result = StringVar()
    result.set("Your Turn!")
    w = Label(root, textvariable=result, fg = "brown")
    w.pack(side=BOTTOM)
    create_frames(root)
    choices = StringVar(root)
    choices.set("Random")
    menu = OptionMenu(root, choices, "Random", "MinMax", "AlphaBeta", "Expectimax")
    menu.pack(side=TOP) 

    root.mainloop()
