import matplotlib.pyplot as plt
import streamlit as st

class Plot:
    def __init__(self):
        pass

    def ploting(self, data=None, scatter=None, title=None, xlabel=None, ylabel=None, figsize=(16, 8), figure=True,
                axhline=None):
        '''
        type data: list
        type scatter: dict
        type title: str
        type xlabel: str
        type ylabel: str

        data:
        [
            {
                list:[...],
                label: '...',
                alpha: ...
            }
        ]

        scatter:
        [
            {
                index: [...],
                list: [...],
                color: '...',
                label: '...',
                marker: '...'
            }
        ]


        axhline:
        [
            {
                num: ...,
                linestyle: ...,
                color: '...',
                alpha: ...
            }
        ]
        '''
        if figure:
            plt.figure(figsize=figsize)

        if data != None:
            for i in range(len(data)):
                plt.plot(data[i]['list'], label=data[i]['label'], alpha=data[i]['alpha'])

        if scatter != None:
            for i in range(len(scatter)):
                plt.scatter(scatter[i]['index'], scatter[i]['list'], scatter[i]['color'], scatter[i]['label'],
                            scatter[i]['marker'])

        if axhline != None:
            for i in range(len(axhline)):
                plt.axhline(axhline[i]['num'], axhline[i]['linestyle'], axhline[i]['color'], axhline[i]['alpha'])

        if title != None:
            plt.title(title)

        if xlabel != None:
            plt.xlabel(xlabel)

        if ylabel != None:
            plt.ylabel(ylabel)

        st.pyplot()