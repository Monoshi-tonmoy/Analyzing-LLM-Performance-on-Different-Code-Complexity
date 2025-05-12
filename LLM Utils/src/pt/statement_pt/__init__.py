from ..abst_pt import AbstPt


class StatementPt1(AbstPt):
    def __init__(self, name, demos=None, args=None):
        super().__init__(name, demos or [])
        self.args = args
        self.demos = demos or []
        
        self.pt_input = ("Given the following {lang} code snippet and output of the code, "
            "what will be the input of the code given the output value?\n\n"
            "Code Snippet\n"
            "```{lang}\n"
            "{code}\n"
            "```\n\n"
            "Function Output:\n"
            "{output}\n\n"
            "Please put your answer in the <ans></ans> tags")
        
        self.pt_output = (
            "Given the following {lang} code snippet and input of the code, "
            "what will be the output of the code given the input value?\n\n"
            "Code Snippet\n"
            "```{lang}\n"
            "{code}\n"
            "```\n\n"
            "Function Inputs:\n"
            "{input}\n\n"
            "Please put your answer in the <ans></ans> tags"
        )

    def task2msg(self, task):
        pt = self.task2pt(task)
        msg = self._pt2msg(pt)
        return msg

    def task2pt(self, task: dict):
        if self.args.prediction == "output":
            pt = self.pt_output.format( 
                lang=self.args.language.lower(),
                code=task['code'],
                input=task['input']
            )
        elif self.args.prediction == "input":
            pt = self.pt_input.format(  
                lang=self.args.language.lower(),
                code=task['code'],
                output=task['output']
            )
        else:
            raise ValueError(f"Invalid prediction type: {self.args.prediction}")
        return pt

    def extract_ans(self, prompt_str, llm_output_str):
        return self.extract_data(llm_output_str, tag_name="ans")

    def _pt2msg(self, pt):
        return [{"content": pt, "role": "user"}]  

    def msg2pt(self):
        raise NotImplementedError("msg2pt not implemented for StatementPt1")


class StatementPt2(StatementPt1):
    def __init__(self, name, demos):
        super().__init__(name, demos)

    def task2pt(self, task: dict ):
        raise NotImplementedError

class StatementPt3(StatementPt1):
    def __init__(self, name, demos):
        super().__init__(name, demos)

    def task2pt(self, task: dict):
        raise NotImplementedError