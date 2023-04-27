import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = f"cmagganas/CoverLetter-GenAI-adapter"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    load_in_8bit=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)


def make_inference(my_cover_letter, job_posting):
    prompt = f"Adapt my Cover Letter ```\n{my_cover_letter}\n```\nto this Job Posting\n```\n{job_posting}\n```"
    batch = tokenizer(
        prompt,
        return_tensors="pt",
    )

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=300)

    return tokenizer.decode(output_tokens[0], skip_special_tokens=True).replace(prompt, '')


if __name__ == "__main__":
    # make a gradio interface
    import gradio as gr

    gr.Interface(
        make_inference,
        [
            gr.inputs.Textbox(lines=5, label="My Cover Letter"),
            gr.inputs.Textbox(lines=10, label="Job Posting")
        ],
        gr.outputs.Textbox(label="Cover Letter"),
        examples=[[
            """
            Dear Hiring Manager,
            I’m excited to be applying for the position at your company. Most recently I was a member of the Cisco Data Science team. I believe my prior experiences, Machine Learning Engineering education from FourthBrain, recent Google Cloud Certification for Professional Machine Learning Engineer and positive open-minded attitude make me an ideal candidate.
            
            As a member of the Cisco CX BCS Data Science team, I led the initiative to gain observability of the ML lifecycle through ML Ops Continuous Monitoring principles as we migrated to Google Cloud. This was a novel project in a relatively new field. It involved working across multiple departments to align everyone’s needs with business goals. I also created an architectural framework and a how-to-implement industry-best-practice road-map. I helped identify many gaps in the monitoring framework and implemented a dashboard with ML lifecycle performance, stability and operational metrics.
            
            During FourthBrain's Machine Learning Engineer Program, I gained the technical and practical skills necessary to contribute to high-performing AI product teams by building, packaging, and deploying state-of-the-art ML models as containerized web applications in cloud-based production environments. The program was divided into four pillars: Data Centric AI, Machine Learning Modeling, AI Applications, and MLOps. As part of our capstone project, my partner and I built a low-latency end-to-end few-shot keyword spotting (FS-KWS) pipeline for personalization running in real-time on an edge device. We presented our project to potential users, collaborators, employers, and the wider open-source ML community during the demo day as part of graduation.
            
            While working as a Data Analyst for HCL, I collected data, wrote ETL scripts and created BI Dashboards to solve business challenges using Google Cloud technologies. I used Data Studio to convey results of these analyses and told a story to emphasize their importance. With the exponential growth of data, knowledge of cloud-based Big Data platforms are integral for solving real-world problems.
            
            While working as a Data Analyst at Commercial Energy, I used VBA in Excel and SQL to analyze customer usage data, using forecasting tools and performing complex calculations to create savings recommendations. I analyzed the price volatility of wholesale natural gas, reporting it to our Chief Risk Officer and the Risk Management Team to make purchasing decisions; a daily process I was able to reduce from two hours to fifteen minutes. I was able to automate and minimize the time and effort it took to complete each task I was responsible for by researching pertinent information and learning new skills. The urgency of the task, my eagerness to prove to myself, and the passion I have for problem solving were my strongest sources of motivation. It taught me that I thrive when I am put to the test and given responsibility.
            
            During my time at Springboard, I learned Machine Learning with Python specifically Natural Language Processing (NLP). I built an application using Twitter data to predict users by class as well as other projects throughout the course. I have a strong background in the hard sciences (Math, Physics and a BS in Actuarial Science) from the University of California, Santa Barbara. I try to apply a data-driven approach to all aspects of my work and hope to do the same with new challenges.
            
            Thank you for your time and consideration.
            I am eager to learn more about this position and demonstrate my skills and fitness.
            
            Sincerely, 
            Christos Magganas
            """,
            """
            Job Title: Machine Learning Engineer
            
            Company: FourthBrain
            
            Location: Remote
            
            Job Description:
            FourthBrain is seeking a highly motivated and skilled Machine Learning Engineer to join our team. The successful candidate will have a strong background in machine learning and software development, with experience building and deploying ML models as containerized web applications in cloud-based production environments.
            
            Responsibilities:
            Collaborate with cross-functional teams to identify opportunities to leverage machine learning for product development or R&D projects.
            Build, optimize, package, and deploy state-of-the-art ML models as containerized web applications in cloud-based production environments.
            Design and develop software solutions that integrate with machine learning models.
            Participate in group capstone projects to demonstrate understanding of MLE software development and its implications.
            Present and share completed projects with potential users, collaborators, employers, or the wider open-source ML community.
            Connect with professionals and employers via guest speaking events and the final project presentation day.
            
            Requirements:
            Bachelor's or Master's degree in computer science, engineering, or a related field.
            Strong proficiency in machine learning algorithms and software development.
            Experience building and deploying ML models as containerized web applications in cloud-based production environments.
            Familiarity with software development tools and practices, such as Git, Linux, and containerization.
            Excellent problem-solving and analytical skills.
            Strong written and verbal communication skills.
            
            At FourthBrain, we value open collaboration, communication, and lifelong learning.
            """
        ]],
        title="CoverLetter-GenAI-adapter",
        description="Write a cover letter for you based on job description.",
    ).launch()
