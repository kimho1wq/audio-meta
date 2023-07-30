#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import print_function
from contextlib import closing
from pycuve.reader import CuveReader
from pycuve.thrift.protocol import CuveKey
import xmltodict
import json

from src.database.data import TrackInfo

__all__ = [
    'Pycuve'
]


class Pycuve:

    #
    location = "korea_real"
    region = "korea"
    group = "naver_music"
    message_type = "track"
    ticket = ""

    meta_list = {'isrc': 'isrc',
                 'title': 'trackTitle',
                 'artist': 'artistNameList',
                 'album': 'albumTitle',
                 'genre': 'genreName'}

    def __init__(self, info_type):
        self.cuve_reader = CuveReader.newBuilder(self.location, self.ticket).build()
        self.info_type = info_type

    def get_content_data(self, trackId: int):
        unique_key = "" + str(trackId)

        try:
            cuve_key = CuveKey().setRegion(self.region).setGroup(self.group).setMessageType(self.message_type)
            cuve_key = cuve_key.setUniqueKey(unique_key)

            document = self.cuve_reader.get(cuve_key)
            content_xml = document.getContentData('internalContentInfo')

            parsed = xmltodict.parse(content_xml)
            content = json.loads(json.dumps(parsed))['content-info']
        except:
            print(f'Invalid trackId : {trackId}')
            return False

        if 'isrc' not in content.keys() or content['isrc'] is None:
            return False

        _dict = {'trackId' : trackId}
        for key in self.meta_list.keys():
            if key in content.keys():
                _dict[self.meta_list[key]] = content[key]
            else:
                _dict[self.meta_list[key]] = None

        return TrackInfo.INFO[self.info_type](_dict)

    def close(self):
        self.cuve_reader = closing(self.cuve_reader)
