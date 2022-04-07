import tkinter as tk
from PIL import Image, ImageTk
import time

window = tk.Tk()

window.title('Lane detection')
window.geometry('600x800')
lbl_1 = tk.Label(window, text='Hello World', bg='yellow', fg='#263238', font=('Arial', 12))
lbl_1.grid(column=0, row=0)



def create_label_image():
    img = Image.open('./top_company_name.jpg')                    
# 讀取圖片
    # img = img.resize( (img.width // 10, img.height // 10) )   
# 縮小圖片

    imgTk =  ImageTk.PhotoImage(img)                        
# 轉換成Tkinter可以用的圖片

    lbl_2 = tk.Label(window, image=imgTk)                   
# 宣告標籤並且設定圖片

    lbl_2.image = imgTk
    lbl_2.grid(column=0, row=0)                             
# 排版位置

def define_layout(obj, cols=1, rows=1):
    
    def method(trg, col, row):
        
        for c in range(cols):    
            trg.columnconfigure(c, weight=1)
        for r in range(rows):
            trg.rowconfigure(r, weight=1)

    if type(obj)==list:        
        [ method(trg, cols, rows) for trg in obj ]
    else:
        trg = obj
        method(trg, cols, rows)



align_mode = 'nswe'
pad = 5

div_size = 200
img_size = div_size * 2
div1 = tk.Frame(window,  width=img_size , height=img_size , bg='white')
div2 = tk.Frame(window,  width=div_size , height=div_size , bg='orange')
div3 = tk.Frame(window,  width=div_size , height=div_size , bg='green')

window.update()
win_size = min( window.winfo_width(), window.winfo_height())
print(win_size)

div1.grid(column=0, row=0, padx=pad, pady=pad, rowspan=2, sticky=align_mode)
div2.grid(column=1, row=0, padx=pad, pady=pad, sticky=align_mode)
div3.grid(column=1, row=1, padx=pad, pady=pad, sticky=align_mode)

define_layout(window, cols=2, rows=2)
define_layout([div1, div2, div3])

# im = Image.open('./top_company_name.jpg')
# imTK = ImageTk.PhotoImage( im.resize( (img_size, img_size) ) )

# image_main = tk.Label(div1, image=imTK)
# image_main['height'] = img_size
# image_main['width'] = img_size

# image_main.grid(column=0, row=0, sticky=align_mode)
seconds = time.time()
local_time = time.ctime(seconds)

lbl_title1 = tk.Label(div2, text=local_time, bg='orange', fg='white',font=('Arial', 20))
lbl_title2 = tk.Label(div2, text="warning system", bg='green', fg='white',font=('Arial', 20))

lbl_title1.grid(column=0, row=0, sticky=align_mode)
lbl_title2.grid(column=0, row=1, sticky=align_mode)

bt1 = tk.Button(div3, text='Start camera', bg='gray', fg='white',font=('Arial', 20))
bt2 = tk.Button(div3, text='Stop', bg='gray', fg='white',font=('Arial', 20))

bt1.grid(column=0, row=0, sticky=align_mode)
bt2.grid(column=0, row=1, sticky=align_mode)

bt1['command'] = lambda : get_size(window, image_main, im)

define_layout(window, cols=2, rows=2)
define_layout(div1)
define_layout(div2, rows=2)
define_layout(div3, rows=4)


create_label_image()

window.mainloop()