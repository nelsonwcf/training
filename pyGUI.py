#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wx
import ctypes

ctypes.windll.shcore.SetProcessDpiAwareness(True)


class QuickImpactApp(wx.App):

    def OnInit(self):
        frame = wx.Frame(None, -1)
        frame.Show(True)
        return True


def main():
    qia = QuickImpactApp()
    qia.MainLoop()


if __name__ == '__main__':
    main()
