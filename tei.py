from bs4 import BeautifulSoup
import nltk
import json
from rouge_score import rouge_scorer
from dataclasses import dataclass

@dataclass
class Person:
    firstname: str
    middlename: str
    surname: str


def elem_to_text(elem, default=''):
    if elem:
        return elem.getText()
    else:
        return default


class TEIFile:
    def __init__(self, data):
        self.soup = BeautifulSoup(data, 'xml')
        self._text = None
        self._title = ''
        self._abstract = ''
        self._headers = None
        self._figures = None

    @property
    def doi(self):
        idno_elem = self.soup.find('idno', type='DOI')
        if not idno_elem:
            return ''
        else:
            return idno_elem.getText()

    @property
    def title(self):
        if not self._title:
            self._title = self.soup.title.getText()
        return self._title

    @property
    def abstract(self):
        if not self._abstract:
            abstract = self.soup.abstract.getText(separator=' ', strip=True)
            self._abstract = abstract
        return self._abstract

    @property
    def authors(self):
        authors_in_header = self.soup.analytic.find_all('author')

        result = []
        for author in authors_in_header:
            persname = author.persname
            if not persname:
                continue
            firstname = elem_to_text(persname.find("forename", type="first"))
            middlename = elem_to_text(persname.find("forename", type="middle"))
            surname = elem_to_text(persname.surname)
            person = Person(firstname, middlename, surname)
            result.append(person)
        return result

    @property
    def text(self):
        if not self._text:
            divs_text = []
            for div in self.soup.body.find_all("div"):
                # div is neither an appendix nor references, just plain text.
                if not div.get("type"):
                    div_text = div.get_text(separator=' ', strip=True)
                    divs_text.append(div_text)
            plain_text = " ".join(divs_text)
            self._text = [{'id': i, 'string': s} for i, s in enumerate(
                nltk.tokenize.sent_tokenize(plain_text))]
        return self._text

    @property
    def headers(self):
        if not self._headers:
            self._headers = []
            scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=False)
            tmptxt = []
            for head in self.soup.find_all("head"):
                if head.get("n"):
                    # print(head)
                    check = head.parent.get_text(
                        separator=' ', strip=True).replace(
                        head.text, "", 1)
                    if check == '':
                        continue
                    # , 'content':nltk.tokenize.sent_tokenize(check)})
                    self._headers.append(
                        {'section': head.text, 'n': head.get('n')})
                    tmptxt.append(check)

            for i, elem in enumerate(self._headers):
                split = nltk.tokenize.sent_tokenize(tmptxt[i])
                if len(split) == 0:
                    print("empty section")
                    continue
                highrouge = 0
                for j in range(len(self.text) - 3):
                    paragraph = '\n'.join([sent['string']
                                           for sent in self._text[j:j + 3]])
                    scores = scorer.score(paragraph, '\n'.join(split[:3]))
                    wtd = scores['rougeLsum'].fmeasure
                    if wtd > highrouge:
                        highrouge = wtd
                        start = j
                if start == -1:
                    print("ERROR:")
                self._headers[i]['start'] = start
            for i in range(len(self._headers) - 1):
                self._headers[i]['end'] = self._headers[i + 1]['start'] - 1
            if len(self._headers) != 0:
                self._headers[-1]['end'] = self._text[-1]['id']

            oldstart = -1
            for i in self._headers:
                if i['start'] < oldstart:
                    print("BIG ERROR:", self.filename)
                    break
                oldstart = i['start']

        return self._headers

    @property
    def figures(self):
        # if not self._figures:
        #     base_name = basename_without_ext(self.filename)
        #     words = ['fig', 'figure', 'table']
        #     self._figures = []
        #     fn = 'figure/data/' + base_name + '.json'
        #     if not path.isfile(fn):
        #         return []
        #     with open(fn) as f:
        #         data = json.load(f)
        #     for i in data:
        #         elem = {
        #             'filename': i['renderURL'],
        #             'caption': i['caption'],
        #             'page': i['page'],
        #             'bbox': i['regionBoundary']}
        #         self._figures.append(elem)
        return self._figures
