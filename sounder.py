#!/usr/bin/python
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

import gtk

import gobject
import pygst
pygst.require("0.10")
import gst
import time
import pango
import struct
import wave
import numpy
import scipy.io
from math import sqrt
from numpy import conj
from numpy.fft import fft, ifft
from time import strftime

# Unpack data from gstreamer
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
#        self.rxpipeline.set_state(gst.STATE_PAUSED)
#        self.txpipeline.set_state(gst.STATE_PAUSED)
        gtk.main_quit()
        
    def update_display(self,event):
        if self.updating:
            return True

        self.updating=True
        
        # Update the data area
        self.dataBlock=numpy.concatenate((self.dataBlock[self.blockSize/2:],self.data))

        # Correlate against chirp reference
        if self.matchedcheck.get_active():
            self.dataBlock=ifft(fft(self.dataBlock)*self.ref*4/self.blockSize**2);

        # Color state
        style=self.screen.get_style()
        gc=style.fg_gc[gtk.STATE_NORMAL]
        gc_oldfg = gc.foreground

        # Erase current display
        gc.foreground=self.black
        self.pixmap.draw_rectangle(gc,True,0,0,self.screenWidth,self.screenHeight)

        gc.foreground = self.white

        # Trigger if desired
        if self.centercheck.get_active():
            idx=numpy.argmax(abs(self.dataBlock))
        else:
            idx=0

        # The following block of code should take the transmitted signal's clock for synchronizing the received signal.  But for some reason, it seems wildly inaccurate (roughly 0.017 s / pulse)
#        maxRange=44 # meters
#        pri=2*maxRange/340.0
#        pos=self.captureTime/1e9-pri/2
#        print(pos)
#        prinum=numpy.floor(pos/pri)
#        idx=round((pos-prinum*pri)*self.sampleRate)
#        print(prinum)
#        print(idx)
#        
#        if idx>self.blockSize/2*self.blocks or idx<0:
#            idx=0

        # Align and downsample pulses
        if self.averagecheck.get_active():
            self.corr_data=numpy.roll(self.corr_data,1,axis=1)
            #self.corr_data[0:numpy.size(self.corr_data,0)-idx,0]=abs(self.dataBlock[idx:])
            self.corr_data[:,0]=numpy.roll(abs(self.dataBlock),-idx,axis=0)
            plotdat=numpy.mean(self.corr_data[::self.get_step(),:],1)
        else:
            #plotdat=self.dataBlock[idx::self.get_step()]
            self.corr_data=numpy.roll(self.dataBlock,-idx,axis=0)
            plotdat=self.corr_data[::self.get_step()]

        # Convert to dB and crop        
        plotdat[plotdat==0]=1e-10
        dbdat=numpy.log10(abs(plotdat))
        data=60*dbdat+100-60*numpy.mean(dbdat)
        data[data<-10]=-10
        data[data>500]=500
        data=data+20
        self.pixmap.draw_lines(gc,[(i2,int(self.screenHeight-e)) for i2,e in enumerate(data)])
        
        # Text labels
        for i in range(1,int(self.screenWidth/50)):
            y=i*50
            rangeMarker=self.pixels_to_cm(y)
            layout=self.screen.create_pango_layout('%0.0f' % rangeMarker + ' cm')
            gc.foreground=self.white
            self.pixmap.draw_layout(gc,y,i*20,layout)
            gc.foreground=self.green
            self.pixmap.draw_line(gc,y,0,y,self.screenHeight)

        # Restore default color
        gc.foreground=gc_oldfg

        # Update the window
        self.screen.window.draw_drawable(self.screen.get_style().fg_gc[gtk.STATE_NORMAL],
                                         self.pixmap, 0, 0, 0, 0, self.screenWidth, self.screenHeight)

        self.updating=False
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
    def new_decoded_cb(self,bin,pad,islast):
        decode=pad.get_parent()
        pipeline=decode.get_parent()
        convert=pipeline.get_by_name("ac2")
        decode.link(convert)
        pipeline.set_state(gst.STATE_PLAYING)

    # Restart the txpipeline after on bank of pulses has been transmitted
    def tx_cb(self,bus,message):
        t=message.type
        if t==gst.MESSAGE_EOS:
            self.txpipeline.seek_simple(gst.FORMAT_TIME,gst.SEEK_FLAG_FLUSH, 0)
        return True

    # Unpack data from gstreamer
    def buffer_cb(self, buffer, pad, user_data):
        #gtk.gdk.threads_enter()
        #self.captureTime=self.txpipeline.query_position(gst.FORMAT_TIME)[0]
        self.data=numpy.array([x*190.0/2.0**16 for x in unpack(pad)])
        #gtk.gdk.threads_leave()

        return True

    ## UI callbacks
    def average_cb(self,event):
        self.corr_data=[]
        self.corr_data=numpy.zeros((self.blockSize/2*self.blocks,self.averagingWindow))
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
            self.txpipeline.set_state(gst.STATE_PLAYING)
        else:
            self.txpipeline.set_state(gst.STATE_PAUSED)
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
            self.pixmap = gtk.gdk.Pixmap(self.screen.window, self.screenWidth, self.screenHeight)
            self.pixmap.draw_rectangle(self.screen.get_style().black_gc,True,0,0,self.screenWidth,self.screenHeight)
            self.width=data.width
            self.height=data.height
        return True

    def __init__(self):
        self.window = gtk.Window()

        self.updating = False

        # Transform parameters
        self.sampleRate=44100
        self.blockSize=3000*self.sampleRate/44100
        self.blocks=1
        self.dataBlock=numpy.zeros((self.blocks*self.blockSize/2))
        self.averagingWindow=10
        self.corr_data=numpy.zeros((self.blockSize/2*self.blocks,self.averagingWindow))

        # Window boilerplate
        self.window.set_title("Sounder")
        self.window.connect("delete_event",self.delete_event)
        self.window.connect("destroy",self.destroy_event)
        self.window.connect("size-allocate",self.size_allocate_event)
        self.window.set_border_width(5)
        self.width=None
        self.height=None
        self.window.set_geometry_hints(min_width=10,min_height=10)

        # Display area boilerplate
        self.screen=gtk.DrawingArea()
        self.screenWidth=512
        self.screenHeight=380
        self.screen.set_size_request(self.screenWidth,self.screenHeight)
        self.zoom=1

        # Load chirp reference
        f=wave.open("squeak.wav",'r')
        wavdata=unpack(f.readframes(f.getnframes()),f.getsampwidth())
        self.ref=numpy.conjugate(fft(wavdata,self.blockSize/2*self.blocks))

        # Construct gstreamer receiver pipeline to funnel data into the application
        # pulsesrc ! audioconvert ! fakesink ! (this program)
        self.rxpipeline=gst.Pipeline("rxpipeline")

        rxsrc=gst.element_factory_make("pulsesrc", "src")
        self.rxpipeline.add(rxsrc)

        ac=gst.element_factory_make("audioconvert","ac")
        self.rxpipeline.add(ac)

        sink=gst.element_factory_make("fakesink","fakesink")
        self.rxpipeline.add(sink)

        rxsrc.link(ac,gst.caps_from_string("audio/x-raw-int,width=16,signed=true,rate="+ str(self.sampleRate) + ",channels=1"))
        rxsrc.set_property("blocksize",self.blockSize)
        ac.link(sink)
        sink.set_property("signal-handoffs",True)
        sink.connect("handoff",self.buffer_cb)

        # Construct gstreamer pipeline for transmission
        self.txpipeline=gst.Pipeline("txpipeline")
        
        txsrc=gst.element_factory_make("filesrc","src2")
        txsrc.set_property("location","squeaks.wav")
        self.txpipeline.add(txsrc)

        dec=gst.element_factory_make("decodebin","wd")
        self.txpipeline.add(dec)
        txsrc.link(dec)
        dec.connect("new-decoded-pad", self.new_decoded_cb)

        ac2=gst.element_factory_make("audioconvert","ac2")
        self.txpipeline.add(ac2)

        sink2=gst.element_factory_make("pulsesink","out")
        self.txpipeline.add(sink2)
        ac2.link(sink2)

        bus=self.txpipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message",self.tx_cb)

        # Organization on window...
        hbox=gtk.HBox(False,10)
        hbox.pack_start(self.screen,False,True,0)
        
        vbox=gtk.VBox(True,0)
        self.matchedcheck=gtk.CheckButton('Matched filter')
        self.matchedcheck.set_active(True)
        self.matchedcheck.connect('toggled',self.average_cb)
        vbox.pack_start(self.matchedcheck,True,True,0)
        self.averagecheck=gtk.CheckButton('Averaging')
        self.averagecheck.set_active(True)
        self.averagecheck.connect('toggled',self.average_cb)
        vbox.pack_start(self.averagecheck,True,True,0)
        self.transmitcheck=gtk.CheckButton('Transmit')
        self.transmitcheck.set_active(True)
        self.transmitcheck.connect('toggled',self.transmit_cb)
        vbox.pack_start(self.transmitcheck,True,True,0)
        self.centercheck=gtk.CheckButton('Center');
        self.centercheck.set_active(True)
        vbox.pack_start(self.centercheck,True,True,0)

        hbox2=gtk.HBox(True,0)
        button=gtk.Button('+')
        button.connect('clicked',self.avg_up)
        hbox2.pack_start(button,True,True,0)
        button=gtk.Button('-')
        button.connect('clicked',self.avg_dn)
        hbox2.pack_start(button,True,True,0)
        vbox.pack_end(hbox2,True,True,0)

        button=gtk.Button('Save')
        button.connect('clicked',self.saveButton)
        vbox.pack_end(button,True,True,0)

        hbox.pack_end(vbox,True,True,0)

        # Assemble the window
        self.window.add(hbox)
        self.window.show_all()

        # Backing pixmap for the display
        self.pixmap = gtk.gdk.Pixmap(self.screen.window, self.screenWidth, self.screenHeight)
        self.pixmap.draw_rectangle(self.screen.get_style().black_gc,True,0,0,self.screenWidth,self.screenHeight)

        # Reference color data
        self.black=self.screen.window.get_colormap().alloc_color(0,0,0)
        self.white=self.screen.window.get_colormap().alloc_color(0xFFFF,0xFFFF,0xFFFF)
        self.red=self.screen.window.get_colormap().alloc_color(0xFFFF,0,0)
        self.green=self.screen.window.get_colormap().alloc_color(0,0xFFFF,0)
        self.blue=self.screen.window.get_colormap().alloc_color(0,0,0xFFFF)

        # Turn on receiver chain
        self.rxpipeline.set_state(gst.STATE_PLAYING) 

        # Wait to start transmitting until the pipeline is fully assembled
        self.txpipeline.set_state(gst.STATE_PAUSED)  

        self.data=numpy.zeros(self.dataBlock.shape)
        gobject.idle_add(self.update_display,self)
        
        return

    def main(self):
        gtk.main()
        return 0

if __name__ == "__main__":
    gtk.gdk.threads_init()
    sounderob=sounder()
    sounderob.main()
