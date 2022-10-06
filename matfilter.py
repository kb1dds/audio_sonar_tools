#!/usr/bin/env python
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

import cairo
import gi
gi.require_version("Gtk","3.0")
gi.require_version("Gst","1.0")
from gi.repository import GObject, Gtk, Gdk, Gst, GLib

import time
import struct
import wave
import numpy
from math import sqrt
from numpy import conj
from numpy.fft import fft, ifft

# Unpack data 
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
        self.pipeline.set_state(Gst.State.NULL)
        Gtk.main_quit()

    def buffer_cb(self, sink):
        # Unpack and FFT data
        sample=sink.emit('pull-sample')
        buffer=sample.get_buffer()
        self.data=numpy.frombuffer(buffer.extract_dup(0,buffer.get_size()),
                                   dtype=numpy.int16)

        return Gst.FlowReturn.OK

    def trigger_update(self):
        rect=self.screen.get_allocation()
        self.window.get_window().invalidate_rect(rect,True)
        return True
        
    def update_display(self,widget,ctx):

        # Update the data area
        self.dataBlock=numpy.concatenate((self.dataBlock[int(self.blockSize/2):],self.data))
        data_fft=fft(self.dataBlock)        

        # Erase current display
        ctx.set_source_rgb(0,0,0)
        ctx.rectangle(0,0,512,380)
        ctx.fill()

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
                    ctx.set_source_rgb(1,1,1)
                elif i==1:
                    ctx.set_source_rgb(1,0,0)
                elif i==2:
                    ctx.set_source_rgb(0,1,0)
                elif i==3:
                    ctx.set_source_rgb(0,0,1)

                if self.centercheck.get_active():
                    idx=numpy.argmax(abs(corr))
                else:
                    idx=0

                if self.averagecheck.get_active():
                    self.corr_data[i][0:numpy.size(self.corr_data[i],0)-idx]+=corr[idx:]
                else:
                    self.corr_data[i]=corr[idx:]

                plotdat=self.corr_data[i][::int(self.blockSize/2*self.blocks/512)]

                data=20*numpy.log10(0.01+abs(self.corr_data[i]))
                data[data<-20]=-20
                data=data+20

                ctx.new_path()
                ctx.move_to(0,int(380-data[0]))
                for i2,e in enumerate(data):
                    ctx.line_to(i2,int(380-e))
                ctx.stroke()

            ctx.set_source_rgb(1,1,1)
            ctx.new_path()
            ctx.arc(xm,ym,4,0,6.28319)
            ctx.stroke()
            ctx.new_path()
            ctx.move_to(xm,ym)
            ctx.line_to(xm,370)
            ctx.stroke()

        if maxnum >= 0:
            self.detectedText.set_text('Detected filter:' + str(maxnum+1))
        else:
            self.detectedText.set_text('Detected filter: None')

        # Text labels
        for i in range(1,10):
            y=i*50
            rangeMarker=self.blockSize*340/self.sampleRate*self.blocks*100*y/1024;

            ctx.set_source_rgb(1,1,1)
            ctx.move_to(y,i*20)
            ctx.set_font_size(12)
            ctx.select_font_face('Arial', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.show_text('%0.0f' % rangeMarker + ' cm')

            ctx.set_source_rgb(0,1,0)
            ctx.new_path()
            ctx.move_to(y,0)
            ctx.line_to(y,380)
            ctx.stroke()

        # Finalize stored data
        if self.storing:
            self.savedData=numpy.append(self.savedData,newData,0)
            self.storeButton.set_label('Store #' + str(numpy.size(self.savedData,0)+1))
            self.storing=False
            
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
        self.ref[data]=numpy.conjugate(fft(wavdata,int(self.blockSize/2*self.blocks)))

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
        dialog=Gtk.FileChooserDialog(title='Save as...',
                                     parent=self.window,
                                     action=Gtk.FileChooserAction.SAVE)
        dialog.add_button(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL)
        dialog.add_button(Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
        
        response=dialog.run()
        if response==Gtk.ResponseType.OK:
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
            self.corr_data.append(numpy.zeros(int(self.blockSize/2*self.blocks),dtype='complex128'))
        return True

    def __init__(self):
        self.window = Gtk.Window()

        # Transform parameters
        self.blockSize=32768
        self.blocks=1
        self.sampleRate=44100
        self.dataBlock=numpy.zeros(int(self.blocks*self.blockSize/2))
        self.filters=8
        self.corr_data=[]
        for i in range(0,self.filters):
            self.corr_data.append(numpy.zeros(int(self.blockSize/2*self.blocks)))

        # Default reference signals
        self.ref=[]
        for i in range(0,self.filters):
            self.ref.append(numpy.zeros(int(self.blockSize/2*self.blocks)))

        # Window boilerplate
        self.window.set_title("Matched filter bank")
        self.window.connect("delete_event",self.delete_event)
        self.window.connect("destroy",self.destroy_event)
        self.window.set_border_width(5)

        # Display area boilerplate
        self.screen=Gtk.DrawingArea()
        self.screen.connect("draw",self.update_display)

        # Construct gstreamer pipeline to funnel data into the application
        # pulsesrc ! capsfilter ! appsink ! (this program)
        self.pipeline=Gst.Pipeline.new("mypipeline")

        src=Gst.ElementFactory.make("pulsesrc", "src")
        src.set_property("blocksize",self.blockSize)
        self.pipeline.add(src)

        ac=Gst.ElementFactory.make("capsfilter","ac")
        ac.set_property("caps",Gst.caps_from_string("audio/x-raw,format=S16LE,rate="+ str(self.sampleRate) + ",channels=1"))
        self.pipeline.add(ac)

        sink=Gst.ElementFactory.make("appsink","as")
        sink.set_property('max-buffers',20)
        sink.set_property("emit-signals",True)
        sink.set_property("sync",False)
        sink.connect("new-sample",self.buffer_cb)
        self.pipeline.add(sink)

        src.link(ac)
        ac.link(sink)

        # Organization on window...
        hbox=Gtk.HBox(homogeneous=True,spacing=0)
        hbox.pack_start(self.screen,False,True,0)
        
        vbox=Gtk.VBox(homogeneous=True,spacing=10)
        hbox.pack_end(vbox,True,True,0)

        hbox2=Gtk.HBox(homogeneous=True,spacing=0)
        self.averagecheck=Gtk.CheckButton(label='Averaging');
        self.averagecheck.connect('toggled',self.average_cb)
        hbox2.pack_start(self.averagecheck,True,True,0)
        self.centercheck=Gtk.CheckButton(label='Center');
        hbox2.pack_start(self.centercheck,True,True,0)
        self.sinrcheck=Gtk.CheckButton(label='SINR');
        hbox2.pack_start(self.sinrcheck,True,True,0)
        vbox.pack_start(hbox2,True,True,0)
        
        self.detectedText=Gtk.Label(label='Detected filter: None')
        vbox.pack_start(self.detectedText,True,True,0)

        # Datafile controls
        hbox2=Gtk.HBox(homogeneous=True,spacing=0)
        self.storeButton=Gtk.Button(label='Store #1')
        self.storeButton.connect('clicked',self.store_cb)
        hbox2.pack_start(self.storeButton,True,True,0)
        clearButton=Gtk.Button(label='Delete Last')
        clearButton.connect('clicked',self.delete_cb)
        hbox2.pack_start(clearButton,True,True,0)
        saveButton=Gtk.Button(label='Save')
        saveButton.connect('clicked',self.save_cb)
        hbox2.pack_start(saveButton,True,True,0)
        vbox.pack_start(hbox2,True,True,0)
        self.storing=False
        self.savedData=numpy.zeros((0,self.filters))

        # Matched filter reference files
        self.entry=[]
        for i in range(0,self.filters):
            hbox2=Gtk.HBox(homogeneous=True,spacing=0)
            capture=Gtk.Button(label='Capture ' + str(i+1))
            capture.connect('clicked',self.capture_cb,i)
            hbox2.pack_start(capture,True,True,0)

            self.entry.append(Gtk.Entry())
            self.entry[i].connect('activate',self.entry_update,i)
            hbox2.pack_start(self.entry[i],True,True,0)

            vbox.pack_start(hbox2,True,True,0)

        # Assemble the window
        self.window.add(hbox)
        self.window.show_all()

        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            print('Error! Did not start pipeline')

        self.data=numpy.zeros(self.dataBlock.shape)
        GLib.timeout_add(50,self.trigger_update)
        
        return

    def main(self):
        Gtk.main()
        return 0

if __name__ == "__main__":
    Gst.init()
    matfilter=matFilter()
    matfilter.main()
