# import project modules
from src.util import *
import src.constants as const

# import libraries
import os
import numpy as np
import pandas as pd
from IPython.display import display
from top2vec import Top2Vec


class SectorModelling:
    def __init__(self):
        log(f'Initializing SectorModelling...')
        self.config = yaml.safe_load(open('config.yml'))


    # load documents (Item 1 - Business)
    def load_docs(self):
        log(f'Loading input docs for top2vec...')
        docs = load_pkl(os.path.join(const.INTERIM_DATA_PATH, 'sampled_docs.pkl'))
        self.documents = [docs[cik]['item_1'] for cik in docs]
        self.document_ids = list(docs)


    def train_top2vec(self):
        log(f'Training top2vec...')
        self.model = Top2Vec(documents = self.documents, 
                            embedding_model = 'universal-sentence-encoder',
                            document_ids = self.document_ids,
                            workers = -1)
        log(f'Training of top2vec completed')


    def analyze_topics(self):
        log(f'Analyzing topic modelling results...')
        # CIK to topic mapping
        doc_topics_ = self.model.get_documents_topics(doc_ids=self.document_ids)
        doc_topics = pd.DataFrame({'cik':self.document_ids, 'topic':doc_topics_[0], 'topic_words':[', '.join(list(x)) for x in list(doc_topics_[2][:,:10])]})
        log(f'Resulting topics:')
        display(doc_topics)

        # number of topics
        n_topic = self.model.get_num_topics()
        log(f'Number of topics: {n_topic}')

        # topic description
        size, topic_num = self.model.get_topic_sizes()
        topics = pd.DataFrame({'topic_num':topic_num, 'size':size})
        topic_words, word_scores, topic_nums = self.model.get_topics(n_topic)
        df = pd.DataFrame({'topic_num':topic_nums, 'topic_words':[', '.join(list(x)) for x in list(topic_words[:,:10])]})
        topics = topics.merge(df, how='inner', on='topic_num')
        log(f'Topic description:')
        display(topics)

        # topic words visualised
        for i in topics.topic_num:
            self.model.generate_topic_wordcloud(i)

        # save results
        self.doc_topics = doc_topics
        self.topics = topics


    def export(self):
        log(f'Exporting topic modelling results...')
        save_pkl(self.doc_topics, 'doc_topics.pkl')
        save_pkl(self.topics, 'topics.pkl')


