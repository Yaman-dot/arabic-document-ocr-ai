from src.SuryaOCR import SuryaOCR

if __name__ == "__main__":
    surya_ocr = SuryaOCR()
    surya_ocr.load_image(r"data\input\Screenshot 2026-03-27 193051.png")
    surya_ocr.inference()
    flag, avg_conf = surya_ocr.flag_low_conf_paper(use_static_threshold=True)
    if flag:
        print(f"Dynamic Threshold is {surya_ocr.dyn_threshold}")
        print(f"Low confidence page ({avg_conf:.2f}) — send to VLM for review")
    else:
        print(f"Threshold is {surya_ocr.dyn_threshold}")
        print(f"Page is OK ({avg_conf:.2f})")