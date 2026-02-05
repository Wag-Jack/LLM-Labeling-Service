from service_invocations.language_translation.aws_translate import run_aws_translation
from service_invocations.language_translation.google_cloud_translation import run_gc_translation
from service_invocations.language_translation.language_judge import judge_translations
from service_invocations.language_translation.microsoft_translator import run_micro_translation

def run_language_translation(europarl_df):
    # Google Cloud Translation
    print('--- Google Cloud Translation ---')
    gc_trans = run_gc_translation(europarl_df)

    # AWS Translate
    print('--- AWS Translate ---')
    aws_trans = run_aws_translation(europarl_df)

    # Microsoft Translator
    print('--- Microsoft Translator ---')
    micro_trans = run_micro_translation(europarl_df)
    
    # LLM Judging of translation services
    print('--- LLM Judging ---')
    judge_translations(gc_trans, aws_trans, micro_trans, europarl_df)