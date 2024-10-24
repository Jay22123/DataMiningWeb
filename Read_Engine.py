import xml.etree.ElementTree as ET
import json
import os


class Reader():
    def ReadXML(self, filepath):
        abstract_texts = []
        content = ET.iterparse(filepath, events=('start', 'end'))

        for event, elem in content:
            if event == 'end' and elem.tag == 'content':
                text = self.get_text_without_tags(elem)
                abstract_texts.append(text)
                elem.clear()

        combined_text = ' '.join(abstract_texts)
        self._content = combined_text

        return {"Content": self._content, "FilePath": filepath}

    def GetFileNum(self):
        return len(self.files)

    def ReadJASON(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        if isinstance(data, dict):  # 如果是字典，使用 .items()
            for key, value in data.items():
                print(f"{key}: {value}")
        elif isinstance(data, list):  # 如果是列表，直接遍歷
            data = {str(index): item for index, item in enumerate(data)}
        else:
            print("Unknown data type")

         # 提取摘要的部分作為context

    def ReadDocument(self, datapath):
        if os.path.isfile(datapath):
            ext = os.path.splitext(datapath)[1]

            if ext == '.xml':
                datamessage = self.ReadXML(datapath)
            elif ext == '.json':
                self.ReadJASON(datapath)
            else:
                print("暫不支援的格式")
        return datamessage

    def get_text_without_tags(self, element):
        parts = [element.text or '']
        for subelem in element:
            parts.append(self.get_text_without_tags(subelem))
            if subelem.tail:
                parts.append(' ' + subelem.tail.strip() + ' ')
            else:
                parts.append(' ')
        return ''.join(parts).strip()
