"""
SlidingWindowTransformersQA.py: Enables a sliding window approach to process long documents using a HuggingFace 
        Transformer Question-Answering model. Also includes functions to assist in visualizing the results.
Dan Hilgart <dhilgart@gmail.com>

Contains 2 class definitions:
    SliderModel :  This class defines the SliderModel which is typically instantiated once per corpus. It contains the 
        model, the tokenizer, the parameters that define how the window slides, and the questions to be asked.
    SliderDocument : This class is instantiated once for each document and contains the necessary functions to:
        - slice the document into windows for each question
        - batch process those question-window pairs through the SliderModel to classify which tokens in each window 
            answer that window's question
        - track which tokens are included in answer-spans in any of the sliding windows (classify_tokens)
        - return the subsets of text which are included in any answer-spans for each question

Contains 11 functions:
    get_tokenizer(model_name):
        Returns the tokenizer for the given model_name
    print_legend(questions, [Optional]colors):
        Prints the legend: each question's text highlighted in the same color that are used to highlight the answers to 
        that question
    print_doc_with_highlights(text, tokens, token_classes, questions, tokenizer):
        Prints the entire document with each answer span highlighted in a color corresponding to the question it answers
    tokens_to_str(tokens, tokenizer):
        Takes a list of token ids and converts them back to string format using the tokenizer
    identify_distinct_spans(token_classes, question_id):
        For a given question_id, finds contiguous spans of Ones or Trues in the token_classes tensor, returning a list 
        of span_start and span_end token-location tuples for each span
    spans_to_str(tokens, token_classes, tokenizer, question_id):
        For a given question_id, returns a list containing, in str form, each distinct span in the token_classes tensor
    find_all_spans(token_classes, question_ids):
        Finds all spans for every question from the token_classes tensor, returning them as a Dataframe, sorted by span 
        start location.
    token_start_locations(text, tokenizer, tokens = None):
        Determines the character location of the start of each token in the text so that tokens and text indexing can be 
        aligned.
    print_span(span_text, colors, question_id):
        Returns the formatted string to display the span_text with the proper highlighting
    print_post_span(span_row, span_end, sorted_spans, text, token_starts):
        Determines what string should come after the current span, whether it is:
        - An unformatted text string (in the case that the next token is not a part of any answer span)
        - A line break (in the case that there is another span following this one that overlaps with it)
        - All the remaining text (in the case of the last span of the document)
    get_colors(num_questions):
        Returns a list of the formatting characters necessary to highlight different questions in different colors, one 
        color for each question
"""

"""
------------------------------------------------------------------------------------------------------------------------
                                                        Imports
------------------------------------------------------------------------------------------------------------------------
"""
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig

"""
------------------------------------------------------------------------------------------------------------------------
                                                        Classes
------------------------------------------------------------------------------------------------------------------------
"""
class SliderModel(object):
    """
    This class defines the SliderModel which is typically instantiated once per corpus. It contains the model, the 
        tokenizer, the parameters that define how the window slides, and the questions to be asked.

    Attributes
    ----------
    model : Transfomers.AutoModelForQuestionAnswering
        The pre-trained Question-Answering model
    tokenizer : Transformers.AutoTokenizer
        The associated tokenizer for the pre-trained model
    max_batch_size : int
        The maximum number of windows to be fed to the model in a single inference batch
    stride : int
        The number of tokens to slide the model from one window to the next
    questions : list(SliderModel.SliderQuestion objects)
        A list of SliderQuestion (a subclass of SliderModel) objects, 1 for each question to be asked by the model
    max_model_tokens : int
        The maximum number of tokens the model can take as input

    Methods
    -------
    set_questions(questions):
        Takes a list of strings and creates SliderQuestion objects for each, storing them in self.questions
    
    Sub-Classes
    -----------
    SliderQuestion:
        A class to store the text, question length (number of tokens), and the tokens themselves for a given question
    """
    def __init__(self, model_name, max_batch_size, stride, questions = None):
        """
        Instantiates a new slider model object, constructing all the necessary attributes

        Parameters
        ----------
        model_name : str
            the pretrained model name or path of the pretrained model to be loaded. See 
                https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoModelForQuestionAnswering.from_pretrained
        max_batch_size : int
            The maximum number of windows to be fed to the model in a single inference batch
        stride : int
            The number of tokens to slide the model from one window to the next
        questions : list(str), optional
            The list of questions can be provided at instatiation, or later via the set_questions(questions) method
        """
        self.max_batch_size = max_batch_size
        self.stride = stride
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.max_model_tokens = AutoConfig.from_pretrained(model_name).max_position_embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        self.questions = []
        if questions is not None:
            self.set_questions(questions)
    
    def set_questions(self, questions):
        """
        Takes a list of strings and creates SliderQuestion objects for each, storing them in self.questions

        Parameters
        ----------
        questions : list(str)
            The list of questions to be asked by the model
        """
        self.questions = []
        for question in questions:
            self.questions.append(self.SliderQuestion(self.tokenizer, question))
    
    class SliderQuestion(object):
        """
        A class to store the text, question length (number of tokens), and the tokens themselves for a given question

        Attributes
        ----------
        text : str
            The text of the question
        tokens : torch.tensor with shape (1, num_tokens)
            The tokens for the question

        Properties
        ----------
        num_tokens : int
            The length of the question in number of tokens
        """
        def __init__(self, tokenizer, text):
            """
            Instantiates a new slider question object, constructing all the necessary attributes

            Parameters
            ----------
            tokenizer : Transformers.AutoTokenizer
                The associated tokenizer for the pre-trained model
            text : str
                The text of the question
            """
            self.text = text
            self.tokens = tokenizer(text, return_tensors='pt')['input_ids']
        
        @property
        def num_tokens(self):
            """
            num_tokens : int
                The length of the question in number of tokens
            """
            return self.tokens.size()[1]
    
class SliderDocument(object):
    """
    This class is instantiated once for each document and contains the necessary functions to:
        - slice the document into windows for each question
        - batch process those question-window pairs through the SliderModel to classify which tokens in each window 
            answer that window's question
        - track which tokens are included in answer-spans in any of the sliding windows (classify_tokens)
        - return the subsets of text which are included in any answer-spans for each question

    Attributes
    ----------
    slider : SlidingWindowTransformersQA.SliderModel
        A pointer to the slider_model object to be used to classify this document
    text : str
        The text of the document
    tokens : torch.tensor with shape (1, num_tokens)
        The tokens for the whole document
    b_token_in_answer : boolean torch.tensor with shape (num_questions, num_tokens)
        A tensor to track which tokens are part of an answer for each question. Starts filled with False, then as 
        answer-spans are found during batch processing, Trues are written to the tensor for each token in each 
        answer-span
    
    Properties
    ----------
    num_tokens : int
        The length of the document in number of tokens
    token_classes : torch.tensor with shape (num_questions, num_tokens)
        Tensor of token classes by whether the token was part of a span that answered the question (1 column for each 
        question). Class 1 = yes, class 0 = no. 
    
    Methods
    -------
    classify_tokens():
        Processes the entire document to classify all tokens for all questions using batch processing
    filtered_text():
        Returns a list with a dictionary for each question which contains:
            - the question
            - a list of the subsets of the text which are included in any answer-spans for this question
    define_batches():
        Defines a list of the batches necessary to be sent to the slider_model in order to process the entire document. 
        Slices the document into windows for each question and groups together qty=max_batch_size question-window-pair 
        inputs for each batch. 
        
    Sub-Classes
    -----------
    DocumentBatch:
        A class to define and process an individual batch to be sent to the model. Contains a list which collects 
        windows (question+context pairs) as they are identified in the slider document's define_batches() function. Also
        contains the necessary methods to process the batch through the model once all the windows have been collected.
    """
    def __init__(self, slider_model, text):
        """
        Instantiates a new slider document object, constructing all the necessary attributes

        Parameters
        ----------
        slider_model : SlidingWindowTransformersQA.SliderModel
            A pointer to the slider_model object to be used to classify this document
        text : str
            The text of the document
        """
        self.slider = slider_model
        self.text = text
        self.tokens = self.slider.tokenizer(text, return_tensors='pt'
                                           )['input_ids'][:,1:-1] # drop first and last tokens ([CLS] and [SEP])
        self.b_token_in_answer = torch.zeros((len(self.slider.questions), self.num_tokens), dtype=bool)

    @property
    def num_tokens(self):
        """
        num_tokens : int
            The length of the document in number of tokens
        """
        return self.tokens.size()[1]
    
    @property
    def token_classes(self):
        """
        token_classes : torch.tensor with shape (num_questions, num_tokens)
            Tensor of token classes by whether the token was part of a span that answered the question (1 column for each 
            question). Class 1 = yes, class 0 = no.
        """
        return self.b_token_in_answer * 1

    def classify_tokens(self):
        """
        Processes the entire document to classify all tokens for all questions using batch processing
        """
        batches = self.define_batches()

        while len(batches)>0:
            batch = batches.pop(0)
            b_token_in_answer_batch = batch.run_batch()
            self.b_token_in_answer = torch.logical_or(self.b_token_in_answer, b_token_in_answer_batch)

    def filtered_text(self):
        """
        Returns a list with a dictionary for each question which contains:
            - the question
            - a list of the subsets of the text which are included in any answer-spans for this question
        
        Returns
        -------
        filtered_text : list(dicts)
        """
        to_return=[]
        for i, question in enumerate(self.slider.questions):
            to_return.append({'question':question.text,
                              'text segments':spans_to_str(self.tokens, self.token_classes, self.slider.tokenizer, i)
                              })
        return to_return

    def define_batches(self):
        """
        Defines a list of the batches necessary to be sent to the slider_model in order to process the entire document. 
        Slices the document into windows for each question and groups together qty=max_batch_size question-window-pair 
        inputs for each batch.
        
        Returns
        -------
        batches : list(SliderDocument.DocumentBatch objects)
        """
        batches = [self.DocumentBatch(self.slider, self.tokens)]

        for i, question in enumerate(self.slider.questions):
            max_context_tokens = self.slider.max_model_tokens - \
                                 question.num_tokens - 1 # -1 for the [SEP] token that will be added to the end
            num_windows = max(1, -(-(self.num_tokens - max_context_tokens) // self.slider.stride) + 1)

            for j in range(num_windows):
                end = min(j * self.slider.stride + max_context_tokens, self.num_tokens)
                start = max(0, end - max_context_tokens)
                batches[-1].add_window(i, start, end)

                if len(batches[-1].windows) >= self.slider.max_batch_size:
                    batches.append(self.DocumentBatch(self.slider, self.tokens))

        return batches
    
    class DocumentBatch(object):
        """
        A class to define and process an individual batch to be sent to the model. Contains a list which collects 
        windows (question+context pairs) as they are identified in the slider document's define_batches() function. Also
        contains the necessary methods to process the batch through the model once all the windows have been collected.

        Attributes
        ----------
        slider : SlidingWindowTransformersQA.SliderModel
            A pointer to the slider_model object to be used to classify this document
        tokens : torch.tensor with shape (1, num_tokens)
            A pointer to the tokens for the whole document, which are stored in the slider_document object
        windows : list(dicts)
            Accumulates a dict for each window in the batch including:
                'question number' : int
                    the int corresponding to the location of the question in the slider_model.questions list
                'question num tokens' : int
                    the length of this question in number of tokens
                'token start' : int
                    the int corresponding to the location of the start of the window in the document tokens list
                'token end' : int
                    the int corresponding to the location of the end of the window in the document tokens list

        Properties
        ----------
        num_tokens : int
            The length of the document in number of tokens
        
        Methods
        -------
        add_window(question_number, token_start, token_end):
            Adds a new window to self.windows
        run_batch():
            Classifies the tokens for all windows in the batch
        generate_inputs():
            Concatenates the inputs for all windows in the batch such that they can be fed into the model at the same 
            time. Called by run_batch()
        window_num_tokens(window):
            For a given window, returns the length of the combined question + context in number of tokens. Called by 
            generate_inputs()
        """
        def __init__(self, slider_model, tokens):
            """
            Instantiates a new document batch object, constructing all the necessary attributes

            Parameters
            ----------
            slider_model : SlidingWindowTransformersQA.SliderModel
                A pointer to the slider_model object to be used to classify this batch
            tokens : torch.tensor with shape (1, num_tokens)
                A pointer to the tokens for the whole document, which are stored in the slider_document object
            """
            self.slider = slider_model
            self.tokens = tokens
            self.windows = []

        @property
        def num_tokens(self):
            """
            num_tokens : int
                The length of the document in number of tokens
            """
            return self.tokens.size()[1]

        def add_window(self, question_number, token_start, token_end):
            """
            Adds a new window to self.windows

            Parameters
            ----------
            'question number' : int
                the int corresponding to the location of the question in the slider_model.questions list
            'token start' : int
                the int corresponding to the location of the start of the window in the document tokens list
            'token end' : int
                the int corresponding to the location of the end of the window in the document tokens list
            """
            self.windows.append({'question number':question_number,
                                'question num tokens':self.slider.questions[question_number].num_tokens,
                                'token start':token_start,
                                'token end':token_end
                                })

        def run_batch(self):
            """
            Classifies the tokens for all windows in the batch

            Returns
            -------
            b_token_in_answer : boolean torch.tensor with shape (num_questions, num_tokens)
                This tensor tracks which tokens are part of an answer for each question, but only for this batch. Starts 
                filled with Falses, then for any answer-spans that were found in this batch, Trues are written to the 
                tensor for each token in each answer-span. Back in the slider document object, the Trues (not the 
                Falses) from this tensor are written to the slider document's b_token_in_answer tensor.
            """
            b_token_in_answer = torch.zeros((len(self.slider.questions), self.num_tokens), dtype=bool)
            if len(self.windows) > 0:
                inputs = self.generate_inputs()

                results = self.slider.model(**inputs)

                combined_logits = torch.cat([results['start_logits'].unsqueeze(0), results['end_logits'].unsqueeze(0)])
                span_input_ids = torch.max(combined_logits, 2)[1]
                span_input_ids[1] += 1  # need to add 1 to end token of span

                # need to slide the token ids to remove the question tokens from the front of the tensor:
                input_question_lengths = torch.tensor([window['question num tokens'] for window in self.windows],
                                                      dtype=torch.long)
                span_token_ids = torch.max(span_input_ids - input_question_lengths, 
                                           torch.zeros_like(span_input_ids) # if no span is found, the model will point 
                                                                            # to the first token of the question. After 
                                                                            # subtracting the question length, this 
                                                                            # would be negative, so set a floor of zero.
                                          )    
                for i, window in enumerate(self.windows):
                    # if a span was found, span_token_ids of the start token will be > 0
                    if span_token_ids[0, i] > 0:
                        span_token_start = span_token_ids[0, i] + window['token start']
                        span_token_end = span_token_ids[1, i] + window['token start']

                        b_token_in_answer[window['question number'],span_token_start:span_token_end + 1] = True
            return b_token_in_answer

        def generate_inputs(self):
            """
            Concatenates the inputs for all windows in the batch such that they can be fed into the model at the same 
            time. Called by run_batch()

            Returns
            -------
            inputs : dict
                The dict of inputs that the QA model expects containing:
                    'input_ids' : torch.tensor with shape (batch_size, num_tokens)
                        The token ids for each token in each window (including the question tokens, context tokens, and 
                        [CLS], [SEP] tokens)
                    'token_type_ids' : torch.tensor with shape (batch_size, num_tokens)
                        Identifies which tokens belong to the question (one) and which belong to the context (zero)
                    'attention_mask' : torch.tensor with shape (batch_size, num_tokens)
                        Identifies which tokens are only for padding (zero) and should be ignored entirely
            """
            max_tokens = max([self.window_num_tokens(window) for window in self.windows])
            batch_size = len(self.windows)

            inputs = {'input_ids': torch.zeros(batch_size, max_tokens, dtype=torch.long),
                    'token_type_ids': torch.zeros(batch_size, max_tokens, dtype=torch.long),
                    'attention_mask': torch.zeros(batch_size, max_tokens, dtype=torch.long)
                    }

            for i, window in enumerate(self.windows):
                question = self.slider.questions[window['question number']]
                question_tokens = question.tokens
                context_tokens = self.tokens[:, window['token start']:window['token end']]

                question_length = question_tokens.size()[1]
                context_length = context_tokens.size()[1]

                inputs['input_ids'][i, :question_length] = question_tokens.squeeze(0)
                inputs['input_ids'][i, question_length:question_length + context_length] = context_tokens.squeeze(0)

                # add final [SEP] token after context (same as final token of question tokens)
                inputs['input_ids'][i, -1] = question_tokens.squeeze(0)[-1]

                inputs['token_type_ids'][i, question_length:] = 1
                inputs['attention_mask'][i, :question_length + context_length + 1] = 1

            return inputs

        def window_num_tokens(self, window):
            """
            For a given window, returns the length of the combined question + context in number of tokens. Called by 
            generate_inputs()

            Parameters
            ----------
            window : dict
                For this individual window, the dict from self.windows

            Returns
            -------
            window_tokens : int
                the length of the combined question + context in number of tokens including the [CLS] and [SEP] tokens
            """
            num_question_tokens = window['question num tokens'] - 2  # don't count the [CLS] and [SEP] tokens
            num_context_tokens = window['token end'] - window['token start']

            # The three 1's in the next line correspond to [CLS], [SEP], [SEP] tokens, respectively
            window_tokens = 1 + num_question_tokens + 1 + num_context_tokens + 1

            return window_tokens

"""
------------------------------------------------------------------------------------------------------------------------
                                                       Functions
------------------------------------------------------------------------------------------------------------------------
"""
def get_tokenizer(model_name):
    """
    Returns the tokenizer for the given model_name

    Parameters
    ----------
    model_name : str
        the pretrained model name or path of the pretrained model that is/was used by the slider_model

    Returns
    -------
    tokenizer : Transformers.AutoTokenizer
        The associated tokenizer for the pre-trained model
    """
    return AutoTokenizer.from_pretrained(model_name)

def print_legend(questions, colors=None):
    """
    Prints the legend: each question's text highlighted in the same color that are used to highlight the answers to that
    question

    Parameters
    ----------
    questions : list(str)
        the list of questions for which to display the answers
    colors : list(str), optional
        the string codes of the colors for each question. If not provided, get_colors() will be called to populate this
    """
    if colors is None:
        colors = get_colors(len(questions))
    print('*'*100)
    print(47*' '+'LEGEND'+47*' ')
    print('*'*100)
    for i, question in enumerate(questions):
        print(f'\x1b[{colors[i]}m{question}\x1b[0m')
    print('*'*100)
    print('')

def print_doc_with_highlights(text, tokens, token_classes, questions, tokenizer):
    """
    Prints the entire document with each answer span highlighted in a color corresponding to the question it answers

    Parameters
    ----------
    text : str
        The text of the document
    tokens : torch.tensor with shape (1, num_tokens)
        The tokens for the whole document
    token_classes : torch.tensor with shape (num_questions, num_tokens)
        Tensor of token classes by whether the token was part of a span that answered the question (1 column for each 
        question). Can be either boolean or int (zeros-and-ones) dtype. Class 1/True = yes, class 0/False = no.
    questions : list(str)
        the list of questions for which to display the answers
    tokenizer : Transformers.AutoTokenizer
        The associated tokenizer for the pre-trained model
    """
    colors = get_colors(len(questions))
    print_legend(questions, colors)

    # collect all answer-spans for all questions in the document, sorted by start_token location:
    sorted_spans = find_all_spans(token_classes = torch.tensor(list(token_classes)), 
                                  question_ids = [x for x in range(len(questions))]
                                 )
    if sorted_spans.shape[0] == 0:
        # no spans found, so print the whole text as is
        print(text)
    else:
        # gather the start character location for each token in the entire document:
        token_starts = token_start_locations(text, tokenizer, torch.tensor(list(tokens)))

        # begin by populating the to_print string with all text prior to the first span:
        span_row = 0
        next_token = sorted_spans.iloc[span_row]['span start']
        to_print = text[:token_starts[next_token]]

        # loop through all spans:
        while span_row < sorted_spans.shape[0]:
            span_info = sorted_spans.iloc[span_row].to_dict()

            # add the text of this span to to_print:
            to_print += print_span(span_text = text[token_starts[span_info['span start']]:
                                                    token_starts[span_info['span end']+1]],
                                   colors = colors,
                                   question_id = span_info['question id']
                                  )
            # add to to_print the text that comes after this span and before the next:
            to_print += print_post_span(span_row, span_info['span end'], sorted_spans, text, token_starts)
            
            span_row += 1
        print(to_print)

def tokens_to_str(tokens, tokenizer):
    """
    Takes a list of token ids and converts them back to string format using the tokenizer

    Parameters
    ----------
    tokens : torch.tensor(1-dimensional) or list(tokens)
        A list of token ids to be untokenized back into a string
    tokenizer : Transformers.AutoTokenizer
        The associated tokenizer for the pre-trained model
    
    Returns
    -------
    tokens_to_str : str
        The string form of the provided list of tokens
    """
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens))

def identify_distinct_spans(token_classes, question_id):
    """
    For a given question_id, finds contiguous spans of Ones or Trues in the token_classes tensor, returning a list of 
    span_start and span_end token-location tuples for each span

    Parameters
    ----------
    token_classes : torch.tensor with shape (num_questions, num_tokens)
        Tensor of token classes by whether the token was part of a span that answered the question (1 column for each 
        question). Can be either boolean or int (zeros-and-ones) dtype. Class 1/True = yes, class 0/False = no.
    question_id : int
        the int corresponding to the column location of the question in the token_classes tensor

    Returns
    -------
    spans : list((int, int) tuples)
    """
    b_token_in_answer = (token_classes==1)

    spans = []
    b_last_token_in_answer = False
    span_start = 0
    for i in range(b_token_in_answer.size()[1]):
        b_curr_token_in_answer = b_token_in_answer[question_id,i]
        if b_last_token_in_answer != b_curr_token_in_answer:
            if b_last_token_in_answer == False:
                span_start = i
            else:
                span_end = i - 1
                spans.append((span_start, span_end))
        b_last_token_in_answer = b_curr_token_in_answer

    return spans

def spans_to_str(tokens, token_classes, tokenizer, question_id):
    """
    For a given question_id, returns a list containing, in str form, each distinct span in the token_classes tensor
    
    Parameters
    ----------
    tokens : torch.tensor with shape (1, num_tokens)
        The tokens for the whole document
    token_classes : torch.tensor with shape (num_questions, num_tokens)
        Tensor of token classes by whether the token was part of a span that answered the question (1 column for each 
        question). Can be either boolean or int (zeros-and-ones) dtype. Class 1/True = yes, class 0/False = no.
    tokenizer : Transformers.AutoTokenizer
        The associated tokenizer for the pre-trained model
    question_id : int
        the int corresponding to the column location of the question in the token_classes tensor

    Returns
    -------
    text_segments : list(strs)
    """
    spans = identify_distinct_spans(token_classes, question_id)

    text_segments = []
    for span_start, span_end in spans:
        text_segments.append(tokens_to_str(tokens=tokens[0,span_start:span_end+1],
                                           tokenizer=tokenizer)
                                          )

    return text_segments

def find_all_spans(token_classes, question_ids):
    """
    Finds all spans for every question from the token_classes tensor, returning them as a Dataframe, sorted by span 
    start location.

    Parameters
    ----------
    token_classes : torch.tensor with shape (num_questions, num_tokens)
        Tensor of token classes by whether the token was part of a span that answered the question (1 column for each 
        question). Can be either boolean or int (zeros-and-ones) dtype. Class 1/True = yes, class 0/False = no.
    question_ids : list(int)
        the list of ids (the int corresponding to the column location of the question in the token_classes tensor) for 
        each question for which to display the answers

    Returns
    -------
    sorted_spans : pd.DataFrame
        A dataframe with a row for each span found (if any) with columns of:
            'question id' : int
                the int corresponding to the column location of the question in the token_classes tensor
            'answer id' : int
                the int corresponding to the location of this answer within the list of answer spans for this question
            'span start' : int
                the location of the start token within the list of document tokens
            'span end' : int
                the location of the end token within the list of document tokens (this token IS part of the span)
    """
    spans = []

    for i in question_ids:
        question_spans = identify_distinct_spans(token_classes, i)
        for j, span in enumerate(question_spans):
            spans.append({'question id':i,
                          'answer id':j,
                          'span start':span[0],
                          'span end':span[1]
                          })
    
    if len(spans)>0:       
        return pd.DataFrame(spans).sort_values(['span start','span end','question id'],axis=0,ascending=True)
    else:
        return pd.DataFrame(columns=['question id','answer id','span start','span end'])

def token_start_locations(text, tokenizer, tokens = None):
    """
    Determines the character location of the start of each token in the text so that tokens and text indexing can be 
    aligned.

    Parameters
    ----------
    text : str
        The text of the document
    tokenizer : Transformers.AutoTokenizer
        The associated tokenizer for the pre-trained model
    tokens : torch.tensor with shape (1, num_tokens), optional
        The tokens for the whole document. If not provided, will be generated by feeding the text into the tokenizer

    Returns
    -------
    token_starts : list(int)
        a list of the character locations in the text for the start of each token in tokens
    """
    if tokens is None:
        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0,1:-1] # drop first and last tokens ([CLS] and [SEP])
    num_tokens = tokens.size()[0]
    
    token_starts = []
    str_start = 0
    for i in range(num_tokens):
        token_to_find = tokens_to_str(tokens[i:i+1], tokenizer)
        find_index = text[str_start:].lower().find(token_to_find)
        str_start += find_index
        token_starts.append(str_start)
    return token_starts

def print_span(span_text, colors, question_id):
    """
    Returns the formatted string to display the span_text with the proper highlighting

    Parameters
    ----------
    span_text : str
        the text of the span
    colors : list(str)
        the list containing the color-format strings for all questions
    question_id : int
        the int corresponding to the location of the question in the colors list
    
    Returns
    -------
    to_print : str
        the string to be printed for the span including all characters necessary for formatting 
    """
    # start color:
    to_print=f'\x1b[{colors[question_id]}m'
    # add span text
    to_print+=span_text
    # end color:
    to_print+=f'\x1b[0m'

    return to_print

def print_post_span(span_row, span_end, sorted_spans, text, token_starts):
    """
    Determines what string should come after the current span, whether it is:
    - An unformatted text string (in the case that the next token is not a part of any answer span)
    - A line break (in the case that there is another span following this one that overlaps with it)
    - All the remaining text (in the case of the last span of the document)

    Parameters
    ----------
    span_row : int
        The row_id of the current span in the sorted_spans DataFrame
    span_end : int
        the location of the last token of the current span in token_starts
    sorted_spans : pd.DataFrame
        A dataframe with a row for each span found (if any) with columns of:
            'question id' : int
                the int corresponding to the column location of the question in the token_classes tensor
            'answer id' : int
                the int corresponding to the location of this answer within the list of answer spans for this question
            'span start' : int
                the location of the start token within the list of document tokens
            'span end' : int
                the location of the end token within the list of document tokens (this token IS part of the span)
    text : str
        The text of the document
    token_starts : list(int)
        a list of the character locations in the text for the start of each token in the document
    
    Returns
    -------
    to_print : str
        the string to be appended to the to_print string after the current span before the next
    """
    # Is this the last span? If so, add all remaining text to to_print:
    if span_row == sorted_spans.shape[0]-1:
        return text[token_starts[span_end+1]:]
    else:
        # This is not the last span. Check whether this span overlaps the next span:
        next_span_start = sorted_spans.iloc[span_row+1]['span start']
        if next_span_start > span_end:
            # The next span does not overlap this span, so add the non-answer text (between this span and the 
            # next span) to to_print
            return text[token_starts[span_end+1]:token_starts[next_span_start]]
        else:
            # The next span DOES overlap this span, so add a line break before moving on to the next span
            return '\n'

def get_colors(num_questions):
    """
    Returns a list of the formatting characters necessary to highlight different questions in different colors, one 
    color for each question

    Parameters
    ----------
    num_questions : int
        the number of questions for which to return colors
    
    Returns
    -------
    colors : list(str)
        a list with one entry for each question where each entry is the special formatting characters corresponding to a
        unique color for that question
    """
    return [f'6;30;4{x}' for x in range(1,num_questions+1)]