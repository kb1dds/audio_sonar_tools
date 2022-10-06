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
from gi.repository import GObject, Gtk, Gdk, Gst, GLib

import time
import struct
import numpy
from math import sqrt
from numpy import conj
from numpy.fft import fft, ifft

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
        
        self.dataBlock=numpy.concatenate((self.dataBlock[int(self.blockSize/2):],self.data))
        data_fft=fft(self.dataBlock)

        # Erase current display
        ctx.set_source_rgb(0,0,0)
        ctx.rectangle(0,0,512,380)
        ctx.fill()
        if( self.mode == 2 ):
            # Draw markers
            ctx.set_source_rgb(1,0,0)
            ctx.new_path()
            ctx.move_to(self.marker1,0)
            ctx.line_to(self.marker1,380)
            ctx.stroke()

            ctx.set_source_rgb(0,1,0)
            ctx.new_path()
            ctx.move_to(self.marker2,0)
            ctx.line_to(self.marker2,380)
            ctx.stroke()
            
            ctx.set_source_rgb(0,0,1)
            ctx.new_path()
            ctx.move_to(self.marker3,0)
            ctx.line_to(self.marker3,380)
            ctx.stroke()

            # Draw autocorrelation
            ctx.set_source_rgb(1,1,1)

            data=20*numpy.log10(0.01+abs(ifft(data_fft*conj(data_fft))))
            data[data<-20]=-20
            data=data+20
            data=data[range(0,511)]

            # Magnitude readouts
            self.marker1_mag.set_text(str(data[int(self.marker1)])+' dB')
            self.marker2_mag.set_text(str(data[int(self.marker2)])+' dB')
            self.marker3_mag.set_text(str(data[int(self.marker3)])+' dB')

            ctx.new_path()
            ctx.move_to(0,int(380-data[0]))
            for i,e in enumerate(data):
                ctx.line_to(i,int(380-e))
            
            ctx.stroke()
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
            dat = numpy.array(self.spectrogram, dtype=numpy.uint8)
            dat.shape=(dat.shape[0],dat.shape[1],1)
            dat = numpy.concatenate((dat,dat,dat,numpy.zeros_like(dat)),axis=2)
            surface = cairo.ImageSurface.create_for_data(dat,cairo.FORMAT_ARGB32,dat.shape[1],dat.shape[0])
            ctx.set_source_surface(surface,0,0)
            ctx.paint()

        else:
            self.spectrogram=numpy.zeros((380,511))

        if( self.mode == 0 or self.mode == 3 ):
            # Draw markers
            
            ctx.set_source_rgb(1,0,0)
            ctx.new_path()
            ctx.move_to(self.marker1,0)
            ctx.line_to(self.marker1,380)
            ctx.stroke()

            ctx.set_source_rgb(0,1,0)
            ctx.new_path()
            ctx.move_to(self.marker2,0)
            ctx.line_to(self.marker2,380)
            ctx.stroke()
            
            ctx.set_source_rgb(0,0,1)
            ctx.new_path()
            ctx.move_to(self.marker3,0)
            ctx.line_to(self.marker3,380)
            ctx.stroke()


        if( self.mode == 0 ):
            # Draw spectrum
            ctx.set_source_rgb(1,1,1)
            ctx.new_path()
            ctx.move_to(0,int(380-data[0]))
            for i,e in enumerate(data):
                ctx.line_to(i,int(380-e))
            
            ctx.stroke()

        if( self.mode == 1 ):
            # Plot points on a track
            marker1val=int(max(abs(data_fft[self.marker1-5:self.marker1+5])))
            marker2val=int(max(abs(data_fft[self.marker2-5:self.marker2+5])))
            newpt=(marker1val,marker2val)
            self.track.append(newpt)
            scale=scale_track(self.track,512,380)

            ctx.set_source_rgb(1,1,1)
            for x in self.track:
                ctx.move_to(int(x[0]*scale),int(x[1]*scale))
                ctx.close_path()
                ctx.stroke()
            
            ctx.set_source_rgb(0,1,0)
            ctx.new_path()
            ctx.move_to(int(scale*newpt[0])-5,int(scale*newpt[1]))
            ctx.line_to(int(scale*newpt[0])+5,int(scale*newpt[1]))
            ctx.stroke()
            ctx.new_path()
            ctx.move_to(int(scale*newpt[0]),int(scale*newpt[1])-5)
            ctx.line_to(int(scale*newpt[0]),int(scale*newpt[1])+5)
            ctx.stroke()

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
        self.screen.connect("draw",self.update_display)

        self.marker1=100
        self.marker2=200
        self.marker3=300
        self.mode=3
        self.track=[]

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

        # Radio buttons for markers and status readout
        self.button1=Gtk.RadioButton(group=None,label="Marker 1")
        self.button1.set_active(True)
        vbox.pack_start(child=self.button1,expand=True,fill=True,padding=0)

        self.marker1_freq=Gtk.Label(label=str(convert_to_hz(self.marker1,self.sampleRate,self.blockSize)) + " Hz")
        vbox.pack_start(self.marker1_freq,True,True,0)

        self.marker1_mag=Gtk.Label(label='0 dB')
        vbox.pack_start(self.marker1_mag,True,True,0)

        self.button2=Gtk.RadioButton(group=self.button1,label="Marker 2")
        self.button2.set_active(False)
        vbox.pack_start(self.button2,True,True,0)

        self.marker2_freq=Gtk.Label(label=str(convert_to_hz(self.marker2,self.sampleRate,self.blockSize)) + " Hz")
        vbox.pack_start(self.marker2_freq,True,True,0)

        self.marker2_mag=Gtk.Label(label='0 dB')
        vbox.pack_start(self.marker2_mag,True,True,0)

        self.button3=Gtk.RadioButton(group=self.button1,label="Marker 3")
        self.button3.set_active(False)
        vbox.pack_start(self.button3,True,True,0)

        self.marker3_freq=Gtk.Label(label=str(convert_to_hz(self.marker3,self.sampleRate,self.blockSize)) + " Hz")
        vbox.pack_start(self.marker3_freq,True,True,0)

        self.marker3_mag=Gtk.Label(label='0 dB')
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

        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            print('Error! Did not start pipeline')

        self.data=numpy.zeros(self.dataBlock.shape)
        GLib.timeout_add(50,self.trigger_update)
        return

    def main(self):
        Gtk.main()
        return 0

if __name__ == "__main__":
    Gst.init(None)
    gtkspec=gtkSpec()
    gtkspec.main()
