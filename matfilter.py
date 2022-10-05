#!/usr/bin/python
# 
# Interative matched filter bank
#  Uses reference signals from WAV files, or as captured from the environment
#  Saves SNR data on user request

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

import gtk

import gobject
import pygst
pygst.require("0.10")
import pango
import gst
import time
import struct
import wave
import numpy
from math import sqrt
from numpy import conj
from numpy.fft import fft, ifft

# Unpack data from gstreamer
def unpack(str,bytesPerSample=2):
    result=list()
    for i in range(0,len(str)-1,bytesPerSample):
        result.append(struct.unpack('h',str[i:i+bytesPerSample])[0])
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

class matFilter:

    def delete_event(self, event, data=None):
        return False

    def destroy_event(self, data=None):
        self.pipeline.set_state(gst.STATE_NULL)
        gtk.main_quit()

    def buffer_cb(self, buffer, pad, user_data):
        gtk.gdk.threads_enter()

        # Unpack and FFT data
        self.data=numpy.array([x*190.0/2.0**16 for x in unpack(pad)])
        gtk.gdk.threads_leave()

        return True
        
    def update_display(self,event):
        if self.updating:
            return True

        self.updating=True
        gtk.gdk.threads_enter()
        
        # Update the data area
        self.dataBlock=numpy.concatenate((self.dataBlock[self.blockSize/2:],self.data))
        data_fft=fft(self.dataBlock)        

        # Color state
        style=self.screen.get_style()
        gc=style.fg_gc[gtk.STATE_NORMAL]
        gc_oldfg = gc.foreground

        # Erase current display
        gc.foreground=self.black
        self.pixmap.draw_rectangle(gc,True,0,0,512,380)

        # Manage storing of data if storage is queued
        if self.storing:
            newData=numpy.zeros((1,self.filters))

        # Plot each filter position
        maxSNR=0
        maxnum=-1
        for i in range(0,self.filters):
            # Compute filter coefficients
            corr=ifft(data_fft*self.ref[i])

            # Compute filter SNR
            if self.sinrcheck.get_active():
                snr=100*numpy.log10(max(abs(corr)+0.01)/(numpy.std(abs(corr))+numpy.mean(abs(corr)+0.01)))
                ym=int(370-snr)
            else:
                snr=100*numpy.log10(max(abs(corr)+0.01))
                ym=int(370-snr/10.0)
                
            # Compute marker location
            xm=int((i+1)*(512/(self.filters+1)))

            if self.storing:
                newData[0,i]=snr/10.0

            # Detect maximum filter readout
            if snr > maxSNR:
                maxSNR=snr
                maxnum=i

            # Plot
            if( i>=0 and i <=3 ):
                if i==0:
                    gc.foreground=self.white
                elif i==1:
                    gc.foreground=self.red
                elif i==2:
                    gc.foreground=self.green
                elif i==3:
                    gc.foreground=self.blue

                if self.centercheck.get_active():
                    idx=numpy.argmax(abs(corr))
                else:
                    idx=0

                if self.averagecheck.get_active():
                    self.corr_data[i][0:numpy.size(self.corr_data[i],0)-idx]+=corr[idx:]
                else:
                    self.corr_data[i]=corr[idx:]

                plotdat=self.corr_data[i][::self.blockSize/2*self.blocks/512]

                data=20*numpy.log10(0.01+abs(self.corr_data[i]))
                data[data<-20]=-20
                data=data+20
                self.pixmap.draw_lines(gc,[(i2,int(380-e)) for i2,e in enumerate(data)])

            gc.foreground=self.white
            self.pixmap.draw_arc(gc,True,xm-2,ym-2,4,4,0,360*64)
            self.pixmap.draw_line(gc,xm,ym,xm,370)

        if maxnum >= 0:
            self.detectedText.set_text('Detected filter:' + str(maxnum+1))
        else:
            self.detectedText.set_text('Detected filter: None')

        # Text labels
        for i in range(1,10):
            y=i*50
            rangeMarker=self.blockSize*340/self.sampleRate*self.blocks*100*y/1024;
            layout=self.screen.create_pango_layout(str(rangeMarker) + ' cm')
            gc.foreground=self.white
            self.pixmap.draw_layout(gc,y,i*20,layout)
            gc.foreground=self.green
            self.pixmap.draw_line(gc,y,0,y,380)

        # Finalize stored data
        if self.storing:
            self.savedData=numpy.append(self.savedData,newData,0)
            self.storeButton.set_label('Store #' + str(numpy.size(self.savedData,0)+1))
            self.storing=False

        # Restore default color
        gc.foreground=gc_oldfg

        # Update the window
        self.screen.window.draw_drawable(self.screen.get_style().fg_gc[gtk.STATE_NORMAL],
                                         self.pixmap, 0, 0, 0, 0, 512, 380)

        self.updating=False
        gtk.gdk.threads_leave()
        return True

    def entry_update(self,event,data): # data contains the index of the entry box that changed
        # Open reference WAV file
        try:
            f=wave.open(self.entry[data].get_text(),'r')
        except:
            print(self.entry[data].get_text())
            return True # Ignore file errors

        # Unpack file into a numpy array
        wavdata=unpack(f.readframes(f.getnframes()),f.getsampwidth())
        self.ref[data]=numpy.conjugate(fft(wavdata,self.blockSize/2*self.blocks))

        return True

    def store_cb(self,event): # Store a new datapoint
        self.storing=True
        return True

    def delete_cb(self,event): # Delete a datapoint
        self.savedData=self.savedData[0:-1,:]
        self.storeButton.set_label('Store #' + str(1+numpy.size(self.savedData,0)))
        return True

    def save_cb(self,event): # Save the data to file
        # Get file to store
        dialog=gtk.FileChooserDialog('Save as...',self.window,
                                     gtk.FILE_CHOOSER_ACTION_SAVE, 
                                     (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                      gtk.STOCK_SAVE, gtk.RESPONSE_OK))
        response=dialog.run()
        if response==gtk.RESPONSE_OK:
            # Save the data
            numpy.savetxt(dialog.get_filename(),self.savedData,fmt='%2.4f')
            
            # Clear old data
            self.storing=False
            self.savedData=numpy.zeros((0,self.filters))
            self.storeButton.set_label('Store #1')

        dialog.destroy()

        return True

    def capture_cb(self,event,data):
        # Capture the current samples as a reference
        self.ref[data]=numpy.conjugate(fft(self.dataBlock))
        return True

    def average_cb(self,event):
        self.corr_data=[]
        for i in range(0,self.filters):
            self.corr_data.append(numpy.zeros((self.blockSize/2*self.blocks)))
        return True

    def __init__(self):
        self.window = gtk.Window()

        self.updating = False

        # Transform parameters
        self.blockSize=32768
        self.blocks=1
        self.sampleRate=44100
        self.dataBlock=numpy.zeros((self.blocks*self.blockSize/2))
        self.filters=8
        self.corr_data=[]
        for i in range(0,self.filters):
            self.corr_data.append(numpy.zeros((self.blockSize/2*self.blocks)))

        # Default reference signals
        self.ref=[]
        for i in range(0,self.filters):
            self.ref.append(numpy.zeros((self.blockSize/2*self.blocks)))

        # Window boilerplate
        self.window.set_title("Matched filter bank")
        self.window.connect("delete_event",self.delete_event)
        self.window.connect("destroy",self.destroy_event)
        self.window.set_border_width(5)

        # Display area boilerplate
        self.screen=gtk.DrawingArea()

        # Construct gstreamer pipeline to funnel data into the application
        # pulsesrc ! audioconvert ! fakesink ! (this program)
        self.pipeline=gst.Pipeline("mypipeline")

        src=gst.element_factory_make("pulsesrc", "src")
        self.pipeline.add(src)

        ac=gst.element_factory_make("audioconvert","ac")
        self.pipeline.add(ac)

        sink=gst.element_factory_make("fakesink","fakesink")
        self.pipeline.add(sink)

        src.link(ac,gst.caps_from_string("audio/x-raw-int,width=16,signed=true,rate="+ str(self.sampleRate) + ",channels=1"))
        ac.link(sink)
        src.set_property("blocksize",self.blockSize)
        
        sink.set_property("signal-handoffs",True)
        sink.connect("handoff",self.buffer_cb)

        # Organization on window...
        hbox=gtk.HBox(True,0)
        hbox.pack_start(self.screen,False,True,0)
        
        vbox=gtk.VBox(True,10)
        hbox.pack_end(vbox,True,True,0)

        hbox2=gtk.HBox(True,0)
        self.averagecheck=gtk.CheckButton('Averaging');
        self.averagecheck.connect('toggled',self.average_cb)
        hbox2.pack_start(self.averagecheck,True,True,0)
        self.centercheck=gtk.CheckButton('Center');
        hbox2.pack_start(self.centercheck,True,True,0)
        self.sinrcheck=gtk.CheckButton('SINR');
        hbox2.pack_start(self.sinrcheck,True,True,0)
        vbox.pack_start(hbox2,True,True,0)
        
        self.detectedText=gtk.Label('Detected filter: None')
        vbox.pack_start(self.detectedText,True,True,0)

        # Datafile controls
        hbox2=gtk.HBox(True,0)
        self.storeButton=gtk.Button('Store #1')
        self.storeButton.connect('clicked',self.store_cb)
        hbox2.pack_start(self.storeButton,True,True,0)
        clearButton=gtk.Button('Delete Last')
        clearButton.connect('clicked',self.delete_cb)
        hbox2.pack_start(clearButton,True,True,0)
        saveButton=gtk.Button('Save')
        saveButton.connect('clicked',self.save_cb)
        hbox2.pack_start(saveButton,True,True,0)
        vbox.pack_start(hbox2,True,True,0)
        self.storing=False
        self.savedData=numpy.zeros((0,self.filters))

        # Matched filter reference files
        self.entry=[]
        for i in range(0,self.filters):
            hbox2=gtk.HBox(True,0)
            capture=gtk.Button('Capture ' + str(i+1))
            capture.connect('clicked',self.capture_cb,i)
            hbox2.pack_start(capture,True,True,0)

            self.entry.append(gtk.Entry())
            self.entry[i].connect('activate',self.entry_update,i)
            hbox2.pack_start(self.entry[i],True,True,0)

            vbox.pack_start(hbox2,True,True,0)

        # Assemble the window
        self.window.add(hbox)
        self.window.show_all()

        # Reference color data
        self.pixmap = gtk.gdk.Pixmap(self.screen.window, 512, 380)
        self.pixmap.draw_rectangle(self.screen.get_style().black_gc,True,0,0,512,380)

        self.black=self.screen.window.get_colormap().alloc_color(0,0,0)
        self.white=self.screen.window.get_colormap().alloc_color(0xFFFF,0xFFFF,0xFFFF)
        self.red=self.screen.window.get_colormap().alloc_color(0xFFFF,0,0)
        self.green=self.screen.window.get_colormap().alloc_color(0,0xFFFF,0)
        self.blue=self.screen.window.get_colormap().alloc_color(0,0,0xFFFF)

        self.pipeline.set_state(gst.STATE_PLAYING)

        self.data=numpy.zeros(self.dataBlock.shape)
        gobject.idle_add(self.update_display,self)
        
        return

    def main(self):
        gtk.main()
        return 0

if __name__ == "__main__":
    gtk.gdk.threads_init()
    matfilter=matFilter()
    matfilter.main()
