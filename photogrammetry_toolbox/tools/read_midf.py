import os

from tqdm import tqdm


class ParserMidf:
    def __init__(self, filenames_mid):
        filenames_mif = f'{os.path.splitext(filenames_mid)[0]}.mif'
        file_mif = open(filenames_mif, 'r').read()
        file_mid = open(filenames_mid, 'r').read()
        self.file_mif = file_mif
        self.lines_mif = file_mif.split('\n')
        self.lines_mid = file_mid.split('\n')
        self.idx_connection = 0
        self.lines_mif = list(filter(lambda x: x != '',  self.lines_mif))
        self.lines_mid = list(filter(lambda x: x != '',  self.lines_mid))
        self.lines_mid = self.convert_mid()

    def convert_mid(self):
        columns = self.get_columns()
        objects = []
        for line in self.lines_mid:
            object = {}
            for col, val in zip(columns, line.split('\t')):
                if col['type'].startswith('Float'):
                    type = float
                elif col['type'].startswith('Char'):
                    type = str
                else:
                    type = str
                object[col['name']] = type(val)
            objects.append(object)
        return objects

    def get_version(self):
        info = self.get_line_started('VERSION')
        if not info:
            return None
        return int(info[0]['attrs'])

    def get_columns(self):
        columns = []
        info = self.get_line_started("COLUMNS")
        if not info:
            return []
        info = info[0]
        start = info['line_num'] + 1
        finish = start + int(info['attrs'])
        for i in range(start, finish):
            line = self.lines_mif[i]
            if line[0] in ['\t', ' ']:
                line = line[1:]
            line = line.replace('\t', '').split(' ')
            if line[0] in ['\t', ' ', '']:
                line = line[1:]
            col_name = line[0]
            col_type = ''.join(line[1:])

            columns.append({"name": col_name, "type": col_type})
        return columns

    def get_description(self):
        columns = []
        info = self.get_line_started("DESCRIPTION")
        if not info:
            return []
        info = info[0]
        start = info['line_num'] + 1
        finish = start + int(info['attrs'])
        for i in range(start, finish):
            line = self.lines_mif[i]
            if line[0] != '\t':
                raise ValueError("Строка в поле description должна начинаться с табуляции")
            line = line.replace("\t", "").split(' ')
            col_name = line[0]
            col_description = ''.join(line[1:])

            if col_description[0] == '"' or col_description[0] == "'":
                col_description = col_description[1:-1]

            columns.append({"name": col_name, "description": col_description})
        return columns

    def get_coord_sys(self):
        info = self.get_line_started("CoordSys")
        if not info:
            return None
        return info[0]['attrs']

    def get_delimiter(self):
        info = self.get_line_started("DELIMITER")
        if not info:
            return "\t"
        delimiter = info[0]['attrs']
        if delimiter[0] == delimiter[-1] and delimiter[0] in ['\'', '\"']:
            delimiter = delimiter[1:-1]
        return delimiter

    def get_line_started(self, string):
        string = str(string)
        l = len(string)
        return_lines = []
        for i in range(len(self.lines_mif)):
            line = self.lines_mif[i]
            if line[:l].lower() == string.lower():
                attrs = line[l:]
                if attrs and attrs[0] == ' ':
                    attrs = attrs[1:]
                return_lines.append({"line_num": int(i), "attrs": attrs, "line_text": self.lines_mif[i]})
        return return_lines

    def calc_stat(self, geom):
        types = (
        "point", "line", "pline", "region", "arc", "text", "rect", "roundrect", "ellipse", "multipoint", "collection")
        stat = {type: 0 for type in types}
        for obj in geom:
            stat[obj['type']] += 1
        for key, val in stat.items():
            if val != 0:
                print(f'Загружено {val} объектов типа {key}')

    def get_data(self):
        """
        Метод получения данных из файлов
        :return:
        """
        info = self.get_line_started("DATA")
        if not info:
            return None

        info = info[0]
        l = info['line_num'] + 1
        m = len(self.lines_mif)

        geom_objects = []
        types = (
        "point", "line", "pline", "region", "arc", "text", "rect", "roundrect", "ellipse", "multipoint", "collection")

        for line_num in range(l, m):
            line = self.lines_mif[line_num]
            if line.lower().startswith(types):
                geom_objects.append(line_num)

        geom = []
        for l in tqdm(geom_objects):
            line = self.lines_mif[l].lower().split(' ')
            parsers = {
                "point": self.__parsePoint,
                "line": self.__parseLine,
                "pline": self.__parsePline,
                "region": self.__parseRegion,
                "arc": self.__parseArc,
                "text": self.__parseText,
                "rect": self.__parseRect,
                "roundrect": self.__parseRoundrect,
                "ellipse": self.__parseEllipse,
                "multipoint": self.__parseMultipoint,
                "collection": self.__parseCollection,
            }
            try:
                func = parsers[line[0]](l)
            except:
                raise ValueError(f'Не известный формат {line[0]} в строке {l})')
            geom.append(func)
        self.calc_stat(geom)
        return geom

    def __parsePoint(self, line):
        """
        Парсинг для формата данных Point
        :param line: номер строки
        :return:
        """
        point = [float(x) for x in self.lines_mif[line].replace('Point ', '').replace('\t', '').split(' ')]
        if 'Z' in self.lines_mid[0]:
            point.append(self.lines_mid[self.idx_connection]['Z'])
            self.idx_connection += 1
        return {"type": "point", "geom": point}

    def __parseLine(self, line):
        """
        Парсинг для формата данных Line
        :param line: номер строки
        :return:
        """
        point = [float(x) for x in self.lines_mif[line].replace('Line ', '').replace('\t', '').split(' ')]
        point1 = point[:2]
        point2 = point[2:]
        if 'Z' in self.lines_mid[0]:
            z = self.lines_mid[self.idx_connection]['Z']
            point1.append(z)
            point2.append(z)
            self.idx_connection += 1
        point = [point1, point2]
        return {"type": "line", "geom": point}

    def __parsePline(self, line):
        """
        Парсинг для формата данных Multipoint
        :param line: номер строки
        :return:
        """
        line_str = self.lines_mif[line].lower().replace('\t', '').split(' ')
        geom_len = 2
        geometry = [[]]
        if len(line_str) > 1 and line_str[1]:
            geom_len = int(line_str[1])
        line += 1
        for l in range(line, line + geom_len):
            point = [float(x) for x in self.lines_mif[l].replace('\t', '').split(' ')]
            if 'Z' in self.lines_mid[0]:
                point.append(self.lines_mid[self.idx_connection]['Z'])
                self.idx_connection += 1
            geometry[0].append(point)
        return {"type": "pline", "geom": geometry}

    def __parseRegion(self, line):
        """
        Парсинг для формата данных Multipoint
        :param line: номер строки
        :return:
        """
        line_str = self.lines_mif[line].lower().replace('\t', '').split(' ')
        reg_count = int(line_str[1])
        reg_len = None
        geometry = []
        for _ in range(reg_count):
            for l in range(line, len(self.lines_mif)):
                try:
                    reg_len = int(self.lines_mif[l])
                except:
                    pass
                else:
                    line = l + 1
                    break
            if not reg_len:
                return None

            for l in range(line, line + reg_len):
                point = [float(x) for x in self.lines_mif[l].replace('\t', '').split(' ')]
                if 'Z' in self.lines_mid[0]:
                    point.append(self.lines_mid[self.idx_connection]['Z'])
                geometry.append(point)
            if 'Z' in self.lines_mid[0]:
                self.idx_connection += 1
        return {"type": "region", "geom": geometry, "reg_count": reg_count}

    # TODO Доделать остальные форматы данных
    def __parseArc(self, line):
        pass
        return {"type": "arc", "geom": None}

    def __parseText(self, line):
        pass
        return {"type": "text", "geom": None}

    def __parseRect(self, line):
        pass
        return {"type": "rect", "geom": None}

    def __parseRoundrect(self, line):
        pass
        return {"type": "roundrect", "geom": None}

    def __parseEllipse(self, line):
        pass
        return {"type": "ellipse", "geom": None}

    def __parseMultipoint(self, line):
        """
        Парсинг для формата данных Multipoint
        :param line: номер строки
        :return:
        """
        line_str = self.lines_mif[line].lower().replace('\t', '').split(' ')
        point_count = int(line_str[1])
        line += 1
        geometry = []
        for i in range(line, line + point_count):
            point = self.lines_mif[i].replace('\t', '').split(' ')
            geometry.append(point)
        return {"type": "multipoint", "geom": geometry}

    def __parseCollection(self, line):
        """
        Парсинг для формата данных Collection
        :param line:
        :return:
        """
        line_str = self.lines_mif[line].lower().replace('\t', '').split(' ')
        collection_arr = []
        collection_len = int(line_str[1])
        line += 1
        types = (
        "point", "line", "pline", "region", "arc", "text", "rect", "roundrect", "ellipse", "multipoint", "collection")
        for l in range(line, len(self.lines_mif)):
            line_str = self.lines_mif[l].lower().replace('\t', '')
            if line_str.startswith(types):
                collection_arr.append(l)
            if len(collection_arr) >= collection_len:
                break
        parsers = {
            "point": self.__parsePoint,
            "line": self.__parseLine,
            "pline": self.__parsePline,
            "region": self.__parseRegion,
            "arc": self.__parseArc,
            "text": self.__parseText,
            "rect": self.__parseRect,
            "roundrect": self.__parseRoundrect,
            "ellipse": self.__parseEllipse,
            "multipoint": self.__parseMultipoint,
            "collection": self.__parseCollection,
        }
        geom_return = []
        for obj in collection_arr:
            geom_type = self.lines_mif[obj].lower().replace('\t', '').split(' ')[0]
            try:
                func = parsers[geom_type]
            except:
                raise ValueError(f'Не известный формат {line[0]} в строке {l})')
            geom_return.append(func(obj))
        return {"type": "collection", "geom": geom_return}
