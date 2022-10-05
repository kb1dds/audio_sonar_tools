#!/usr/bin/env python
# Audio spectrum analyzer

# Copyright (c) 2011, 2022 Michael Robinson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Version 0.1

import cairo
import gi
gi.require_version("Gtk","3.0")
gi.require_version("Gst","1.0")
from gi.repository import GObject, Gtk, Gdk, Gst

import time
import struct
import numpy
from math import sqrt
from numpy import conj
from numpy.fft import fft, ifft

# Unpack data from gstreamer
def unpack(str):
    result=list()
    for i in range(0,len(str)-1,2):
        result.append(struct.unpack('h',str[i:i+2])[0])
    return result

# Convert x location in plot to Hz
def convert_to_hz(x,sample_rate,block_size):
    return x*sample_rate/block_size*2

def convert_to_time(x,sample_rate):
    return x*1000/sample_rate

# Determine how to set scale bounds for track
def scale_track(track,height,width):
    maxx=sqrt(sum([x[0]**2 for x in track])/len(track))
    maxy=sqrt(sum([x[1]**2 for x in track])/len(track))
    return min([width/maxx,height/maxy])

class gtkSpec:

    def delete_event(self, event, data=None):
        return False

    def destroy_event(self, data=None):
        self.pipeline.set_state(Gst.State.NULL)
        Gtk.main_quit()

    def buffer_cb(self, buffer, pad, user_data):
        Gdk.threads_enter()

        # Unpack and FFT data
        self.data=numpy.array([x*190.0/2.0**16 for x in unpack(pad)])
        Gdk.threads_leave()

        return True
        
    def update_display(self,event):
        if self.updating:
            return False

        self.updating=True
        Gdk.threads_enter()
        
        self.dataBlock=numpy.concatenate((self.dataBlock[self.blockSize/2:],self.data))
        data_fft=fft(self.dataBlock)

        # Color state
        style=self.screen.get_style()
        gc=style.fg_gc[Gtk.State.NORMAL]
        gc_oldfg = gc.foreground

        # Erase current display
        gc.foreground=self.black
        self.pixmap.draw_rectangle(gc,True,0,0,512,380)
        if( self.mode == 2 ):
            # Draw markers
            gc.foreground=self.red
            self.pixmap.draw_line(gc,self.marker1,0,self.marker1,380)
            gc.foreground=self.green
            self.pixmap.draw_line(gc,self.marker2,0,self.marker2,380)
            gc.foreground=self.blue
            self.pixmap.draw_line(gc,self.marker3,0,self.marker3,380)

            # Draw autocorrelation
            gc.foreground=self.white

            data=20*numpy.log10(0.01+abs(ifft(data_fft*conj(data_fft))))
            data[data<-20]=-20
            data=data+20
            data=data[range(0,511)]

            # Magnitude readouts
            self.marker1_mag.set_text(str(data[int(self.marker1)])+' dB')
            self.marker2_mag.set_text(str(data[int(self.marker2)])+' dB')
            self.marker3_mag.set_text(str(data[int(self.marker3)])+' dB')

            self.pixmap.draw_lines(gc,[(i,int(380-e)) for i,e in enumerate(data)])
        if( self.mode == 0 or self.mode == 1 or self.mode == 3): # Frequency domain preproc
            data=20*numpy.log10(0.01+abs(data_fft))

            # Adjust data for better plotting
            data[data<-20]=-20
            data=data+20
            data=data[range(0,511)]

            # Magnitude readouts
            self.marker1_mag.set_text(str(data[int(self.marker1)])+' dB')
            self.marker2_mag.set_text(str(data[int(self.marker2)])+' dB')
            self.marker3_mag.set_text(str(data[int(self.marker3)])+' dB')

        if( self.mode == 3 ):
            self.spectrogram[1:379,:]=self.spectrogram[0:378,:]
            self.spectrogram[0,:]=data*2
            self.pixmap.draw_gray_image(gc,0,0,511,380,Gdk.RGB_DITHER_NONE,self.spectrogram.astype('uint8'),511)
        else:
            self.spectrogram=numpy.zeros((380,511))

        if( self.mode == 0 or self.mode == 3 ):
            # Draw markers
            gc.foreground=self.red
            self.pixmap.draw_line(gc,self.marker1,0,self.marker1,380)
            gc.foreground=self.green
            self.pixmap.draw_line(gc,self.marker2,0,self.marker2,380)
            gc.foreground=self.blue
            self.pixmap.draw_line(gc,self.marker3,0,self.marker3,380)
            gc.foreground=self.white

        if( self.mode == 0 ):
            # Draw spectrum
            gc.foreground=self.white
            self.pixmap.draw_lines(gc,[(i,int(380-e)) for i,e in enumerate(data)])
        if( self.mode == 1 ):
            # Plot points on a track
            marker1val=int(max(abs(data_fft[self.marker1-5:self.marker1+5])))
            marker2val=int(max(abs(data_fft[self.marker2-5:self.marker2+5])))
            newpt=(marker1val,marker2val)
            self.track.append(newpt)
            scale=scale_track(self.track,512,380)
            gc.foreground=self.white
            self.pixmap.draw_points(gc,[(int(x[0]*scale),int(x[1]*scale)) for x in self.track])
            gc.foreground=self.green
            self.pixmap.draw_segments(gc,[(int(scale*newpt[0])-5,int(scale*newpt[1]),int(scale*newpt[0])+5,int(scale*newpt[1])),
                                          (int(scale*newpt[0]),int(scale*newpt[1])-5,int(scale*newpt[0]),int(scale*newpt[1])+5)])

        # Restore default color
        gc.foreground=gc_oldfg

        # Update the window
        self.screen.window.draw_drawable(self.screen.get_style().fg_gc[Gtk.State.NORMAL],
                                         self.pixmap, 0, 0, 0, 0, 512, 380)

        self.updating=False
        Gdk.threads_leave()
        return True

    def swapmode(self,event):
        self.mode=self.modeSelect.get_active()
        self.repaint(self)
        if( self.mode == 2 or self.mode == 0 ):
            self.track=[]
        return True

    def button_cb(self,widget,event):
        if event.button == 1 and (self.mode == 0 or self.mode == 2 or self.mode == 3 ):
            if self.button1.get_active():
                self.marker1=int(event.x)
            elif self.button2.get_active():
                self.marker2=int(event.x)
            else:
                self.marker3=int(event.x)
        self.repaint(self)
        return True

    def repaint(self,event):
        if( self.mode == 2 ):
            self.marker1_freq.set_text(str(convert_to_time(self.marker1,self.sampleRate)) + " ms")
            self.marker2_freq.set_text(str(convert_to_time(self.marker2,self.sampleRate)) + " ms")
            self.marker3_freq.set_text(str(convert_to_time(self.marker3,self.sampleRate)) + " ms")
        else:
            self.marker1_freq.set_text(str(convert_to_hz(self.marker1,self.sampleRate,self.blockSize)) + " Hz")
            self.marker2_freq.set_text(str(convert_to_hz(self.marker2,self.sampleRate,self.blockSize)) + " Hz")
            self.marker3_freq.set_text(str(convert_to_hz(self.marker3,self.sampleRate,self.blockSize)) + " Hz")
        return True

    def __init__(self):
        self.window = Gtk.Window()

        self.updating = False

        # Transform parameters
        self.blockSize=2048
        self.blocks=1
        self.sampleRate=44100
        self.dataBlock=numpy.zeros(int(self.blocks*self.blockSize/2))
        self.spectrogram=numpy.zeros((380,511))

        # Window boilerplate
        self.window.set_title("Python Spectrum Analyzer")
        self.window.connect("delete_event",self.delete_event)
        self.window.connect("destroy",self.destroy_event)
        self.window.set_border_width(5)

        # Display area boilerplate
        self.screen=Gtk.DrawingArea()
        self.screen.set_size_request(512,380)
        self.screen.connect("button_press_event",self.button_cb)
        self.screen.add_events( Gdk.EventMask.BUTTON_PRESS_MASK )
        self.marker1=100
        self.marker2=200
        self.marker3=300
        self.mode=3
        self.track=[]

        # Construct gstreamer pipeline to funnel data into the application
        # pulsesrc ! audioconvert ! fakesink ! (this program)
        self.pipeline=Gst.Pipeline.new("mypipeline")

        src=Gst.ElementFactory.make("pulsesrc", "src")
        self.pipeline.add(src)

        ac=Gst.ElementFactory.make("audioconvert","ac")
        self.pipeline.add(ac)

        sink=Gst.ElementFactory.make("fakesink","fakesink")
        self.pipeline.add(sink)

        src.link(ac)#,Gst.caps_from_string("audio/x-raw-int,width=16,signed=true,rate="+ str(self.sampleRate) + ",channels=1"))
        ac.link(sink)
        src.set_property("blocksize",self.blockSize)
        
        sink.set_property("signal-handoffs",True)
        sink.connect("handoff",self.buffer_cb)

        # Organization on window...
        hbox=Gtk.HBox(True,0)
        hbox.pack_start(self.screen,False,True,0)
        
        vbox=Gtk.VBox(True,10)
        hbox.pack_end(vbox,True,True,0)

        # Radio buttons for markers and status readout
        self.button1=Gtk.RadioButton(None,"Marker 1")
        self.button1.set_active(True)
        vbox.pack_start(self.button1,True,True,0)

        self.marker1_freq=Gtk.Label(str(convert_to_hz(self.marker1,self.sampleRate,self.blockSize)) + " Hz")
        vbox.pack_start(self.marker1_freq,True,True,0)

        self.marker1_mag=Gtk.Label('0 dB')
        vbox.pack_start(self.marker1_mag,True,True,0)

        self.button2=Gtk.RadioButton(self.button1,"Marker 2")
        vbox.pack_start(self.button2,True,True,0)

        self.marker2_freq=Gtk.Label(str(convert_to_hz(self.marker2,self.sampleRate,self.blockSize)) + " Hz")
        vbox.pack_start(self.marker2_freq,True,True,0)

        self.marker2_mag=Gtk.Label('0 dB')
        vbox.pack_start(self.marker2_mag,True,True,0)

        self.button3=Gtk.RadioButton(self.button1,"Marker 3")
        vbox.pack_start(self.button3,True,True,0)

        self.marker3_freq=Gtk.Label(str(convert_to_hz(self.marker3,self.sampleRate,self.blockSize)) + " Hz")
        vbox.pack_start(self.marker3_freq,True,True,0)

        self.marker3_mag=Gtk.Label('0 dB')
        vbox.pack_start(self.marker3_mag,True,True,0)

        self.modeSelect=Gtk.ComboBoxText()
        self.modeSelect.append_text('Spectrum')
        self.modeSelect.append_text('XY track')
        self.modeSelect.append_text('Autocorrelation')
        self.modeSelect.append_text('Spectrogram')
        self.modeSelect.set_active(self.mode)
        self.modeSelect.connect('changed',self.swapmode)
        vbox.pack_end(self.modeSelect,True,True,0)

        self.window.add(hbox)
        self.window.show_all()

        self.pixmap = Gdk.Pixmap(self.screen.window, 512, 380)
        self.pixmap.draw_rectangle(self.screen.get_style().black_gc,True,0,0,512,380)

        self.black=self.screen.window.get_colormap().alloc_color(0,0,0)
        self.white=self.screen.window.get_colormap().alloc_color(0xFFFF,0xFFFF,0xFFFF)
        self.red=self.screen.window.get_colormap().alloc_color(0xFFFF,0,0)
        self.green=self.screen.window.get_colormap().alloc_color(0,0xFFFF,0)
        self.blue=self.screen.window.get_colormap().alloc_color(0,0,0xFFFF)

        self.pipeline.set_state(Gst.State.PLAYING)

        self.data=numpy.zeros(self.dataBlock.shape)
        GObject.idle_add(self.update_display,self)
        return

    def main(self):
        Gtk.main()
        return 0

if __name__ == "__main__":
    Gst.init(None)
    gtkspec=gtkSpec()
    gtkspec.main()
