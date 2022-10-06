#!/usr/bin/env python
#
# 1-d sonar sounder 

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
import scipy.io
from math import sqrt
from numpy import conj
from numpy.fft import fft, ifft
from time import strftime

def unpack(str,bytesPerSample=2):
    result=list()
    for i in range(0,len(str)-1,bytesPerSample):
        result.append(struct.unpack('h',str[i:i+bytesPerSample])[0])
    return result

class sounder:
    def delete_event(self, event, data=None):
        return False

    def destroy_event(self, data=None):
        # Apparently, the application crashes intermittently if I try to clean up 
        # gstreamer nicely.  I think there might be a deadlock since the tx pipeline 
        # gets re-seeked frequently.  In any event, it's apparently mostly OK to 
        # let the gc do the cleanup work for me, though this occasionally segfaults...
        Gtk.main_quit()

    def trigger_update(self):
        rect=self.screen.get_allocation()
        self.window.get_window().invalidate_rect(rect,True)
        return True

    def update_display(self,widget,ctx):
       
        # Update the data area
        self.dataBlock=numpy.concatenate((self.dataBlock[int(self.blockSize/2):],self.data))

        # Correlate against chirp reference
        if self.matchedcheck.get_active():
            self.dataBlock=ifft(fft(self.dataBlock)*self.ref*4/self.blockSize**2);

        # Erase current display
        ctx.set_source_rgb(0,0,0)
        ctx.rectangle(0,0,self.screenWidth,self.screenHeight)
        ctx.fill()

        ctx.set_source_rgb(1,1,1)

        # Trigger if desired
        if self.centercheck.get_active():
            idx=numpy.argmax(abs(self.dataBlock))
        else:
            idx=0

        # Align and downsample pulses
        if self.averagecheck.get_active():
            self.corr_data=numpy.roll(self.corr_data,1,axis=1)
            self.corr_data[:,0]=numpy.roll(abs(self.dataBlock),-idx,axis=0)
            plotdat=numpy.mean(self.corr_data[::self.get_step(),:],1)
        else:
            self.corr_data=numpy.roll(self.dataBlock,-idx,axis=0)
            plotdat=self.corr_data[::self.get_step()]

        # Convert to dB and crop        
        plotdat[plotdat==0]=1e-10
        dbdat=numpy.log10(abs(plotdat))
        data=60*dbdat+100-60*numpy.mean(dbdat)
        data[data<-10]=-10
        data[data>500]=500
        data=data+20

        # Plot the echo data
        ctx.new_path()
        ctx.move_to(0,int(self.screenHeight-data[0]))
        for i2,e in enumerate(data):
            ctx.line_to(i2,int(self.screenHeight-e))
        ctx.stroke()            
        
        # Text labels
        for i in range(1,int(self.screenWidth/50)):
            y=i*50
            rangeMarker=self.pixels_to_cm(y)

            ctx.set_source_rgb(1,1,1)
            ctx.move_to(y,i*20)
            ctx.set_font_size(12)
            ctx.select_font_face('Arial', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.show_text('%0.0f' % rangeMarker + ' cm')
            
            ctx.set_source_rgb(0,1,0)
            ctx.new_path()
            ctx.move_to(y,0)
            ctx.line_to(y,self.screenHeight)
            ctx.stroke()

        return True

    # Set plotting interval (samples/pixel)
    def get_step(self):
        if self.zoom <= 0:
            self.zoom=1
        
        step=round(self.blockSize/2.0*self.blocks/self.screenWidth/self.zoom)
        if step <= 1:
            step=1
            self.zoom=self.zoom-1

        return step

    # Convert pixel locations into monostatic ranges in centimeters
    def pixels_to_cm(self,rangePixels):
        samples2cm=34029.0/self.sampleRate/2 # Divide by 2 for range
        return rangePixels*samples2cm*self.get_step()

    ## GStreamer callbacks
    # Assemble the transmit pipeline as it gets linked together
    def new_decoded_cb(self,element,pad):
        decode=pad.get_parent()
        pipeline=decode.get_parent()
        sink=pipeline.get_by_name("out")
        pad.link(sink.get_static_pad('sink'))

    # Restart the txpipeline after on bank of pulses has been transmitted
    def tx_cb(self,bus,message):
        t=message.type
        if t==Gst.MessageType.EOS:
            self.txpipeline.seek_simple(Gst.Format.TIME,Gst.SeekFlags.FLUSH, 0)
        return True

    # Unpack data from gstreamer
    def buffer_cb(self, sink):
        # Unpack and FFT data
        sample=sink.emit('pull-sample')
        buffer=sample.get_buffer()
        self.data=numpy.frombuffer(buffer.extract_dup(0,buffer.get_size()),
                                   dtype=numpy.int16)

        return Gst.FlowReturn.OK


    ## UI callbacks
    def average_cb(self,event):
        self.corr_data=[]
        self.corr_data=numpy.zeros((int(self.blockSize/2*self.blocks),self.averagingWindow))
        return True

    def avg_up(self,event):
        self.averagingWindow+=10
        self.average_cb(event)
        return True

    def avg_dn(self,event):
        self.averagingWindow-=10
        if(self.averagingWindow < 10):
            self.averagingWindow=10
        self.average_cb(event)
        return True

    def zoom_in(self,event):
        self.zoom=self.zoom+1
        return True

    def zoom_out(self,event):
        self.zoom=self.zoom-1
        return True

    def saveButton(self,event):
        # Obtain date and time
        filename=strftime("%Y%m%d%H%M%S.mat")
        scipy.io.savemat(filename,{'corr_data':self.corr_data,
                                   'blockSize':self.blockSize,
                                   'averagingWindow':self.averagingWindow,
                                   'sampleRate':self.sampleRate,
                                   'blocks':self.blocks})
        return True

    def transmit_cb(self,event):
        if self.transmitcheck.get_active():
            if self.txpipeline.set_state(Gst.State.PLAYING)  == Gst.StateChangeReturn.FAILURE:
                print('oh no!')
        else:
            self.txpipeline.set_state(Gst.State.PAUSED)
        return True

    def size_allocate_event(self,event,data=None):
        if self.width == None or self.height == None:
            self.width=data.width
            self.height=data.height
        else:
            # Rebuild the screen
            self.screenHeight+=data.height-self.height
            self.screenWidth+=data.width-self.width

            self.screen.set_size_request(self.screenWidth,self.screenHeight)
            self.width=data.width
            self.height=data.height
        return True

    def __init__(self):
        self.window = Gtk.Window()

        # Transform parameters
        self.sampleRate=44100
        self.blockSize=3000*self.sampleRate/44100
        self.blocks=1
        self.dataBlock=numpy.zeros(int(self.blocks*self.blockSize/2))
        self.averagingWindow=10
        self.corr_data=numpy.zeros((int(self.blockSize/2*self.blocks),self.averagingWindow))

        # Window boilerplate
        self.window.set_title("Sounder")
        self.window.connect("delete_event",self.delete_event)
        self.window.connect("destroy",self.destroy_event)
        self.window.connect("size-allocate",self.size_allocate_event)
        self.window.set_border_width(5)
        self.width=None
        self.height=None

        # Display area boilerplate
        self.screen=Gtk.DrawingArea()
        self.screenWidth=512
        self.screenHeight=380
        self.screen.set_size_request(self.screenWidth,self.screenHeight)
        self.zoom=1
        self.screen.connect("draw",self.update_display)

        # Load chirp reference
        f=wave.open("squeak.wav",'r')
        wavdata=unpack(f.readframes(f.getnframes()),f.getsampwidth())
        self.ref=numpy.conjugate(fft(wavdata,self.blockSize/2*self.blocks))

        # Construct gstreamer receiver pipeline to funnel data into the application
        # pulsesrc ! capsfilter ! appsink ! (this program)
        self.rxpipeline=Gst.Pipeline.new("rxpipeline")

        rxsrc=Gst.ElementFactory.make("pulsesrc", "src")
        rxsrc.set_property("blocksize",self.blockSize)
        self.rxpipeline.add(rxsrc)

        ac=Gst.ElementFactory.make("capsfilter","ac")
        ac.set_property("caps",Gst.caps_from_string("audio/x-raw,format=S16LE,rate="+ str(self.sampleRate) + ",channels=1"))
        self.rxpipeline.add(ac)

        sink=Gst.ElementFactory.make("appsink","as")
        sink.set_property('max-buffers',20)
        sink.set_property("emit-signals",True)
        sink.set_property("sync",False)
        sink.connect("new-sample",self.buffer_cb)
        self.rxpipeline.add(sink)

        rxsrc.link(ac)
        ac.link(sink)

        # Construct gstreamer pipeline for transmission
        self.txpipeline=Gst.Pipeline.new("txpipeline")
        
        txsrc=Gst.ElementFactory.make("filesrc","txsrc")
        txsrc.set_property("location","squeaks.wav")
        self.txpipeline.add(txsrc)

        wd=Gst.ElementFactory.make("decodebin","wd")
        self.txpipeline.add(wd)
        txsrc.link(wd)
        wd.connect("pad-added", self.new_decoded_cb)

        sink2=Gst.ElementFactory.make("pulsesink","out")
        self.txpipeline.add(sink2)

        bus=self.txpipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message",self.tx_cb)

        # Organization on window...
        hbox=Gtk.HBox(homogeneous=False,spacing=10)
        hbox.pack_start(self.screen,False,True,0)
        
        vbox=Gtk.VBox(homogeneous=True,spacing=0)
        self.matchedcheck=Gtk.CheckButton(label='Matched filter')
        self.matchedcheck.set_active(True)
        self.matchedcheck.connect('toggled',self.average_cb)
        vbox.pack_start(self.matchedcheck,True,True,0)
        self.averagecheck=Gtk.CheckButton(label='Averaging')
        self.averagecheck.set_active(True)
        self.averagecheck.connect('toggled',self.average_cb)
        vbox.pack_start(self.averagecheck,True,True,0)
        self.transmitcheck=Gtk.CheckButton(label='Transmit')
        self.transmitcheck.set_active(False)
        self.transmitcheck.connect('toggled',self.transmit_cb)
        vbox.pack_start(self.transmitcheck,True,True,0)
        self.centercheck=Gtk.CheckButton(label='Center');
        self.centercheck.set_active(True)
        vbox.pack_start(self.centercheck,True,True,0)

        hbox2=Gtk.HBox(homogeneous=True,spacing=0)
        button=Gtk.Button(label='AVG+')
        button.connect('clicked',self.avg_up)
        hbox2.pack_start(button,True,True,0)
        button=Gtk.Button(label='AVG-')
        button.connect('clicked',self.avg_dn)
        hbox2.pack_start(button,True,True,0)
        vbox.pack_end(hbox2,True,True,0)

        button=Gtk.Button(label='Save')
        button.connect('clicked',self.saveButton)
        vbox.pack_end(button,True,True,0)

        hbox.pack_end(vbox,True,True,0)

        # Assemble the window
        self.window.add(hbox)
        self.window.show_all()

        # Turn on receiver chain
        self.rxpipeline.set_state(Gst.State.PLAYING) 

        # Wait to start transmitting until the pipeline is fully assembled
        self.txpipeline.set_state(Gst.State.PAUSED)  

        self.data=numpy.zeros(self.dataBlock.shape)
        GLib.timeout_add(50,self.trigger_update)
        return

    def main(self):
        Gtk.main()
        return 0

if __name__ == "__main__":
    Gst.init(None)
    sounderob=sounder()
    sounderob.main()
