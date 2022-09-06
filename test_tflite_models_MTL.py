import os
import tensorflow as tf

text_output_dir = "./new_profile_text_MTL"
if not os.path.exists(text_output_dir):
    os.makedirs(text_output_dir)


plot_output_dir = "./new_profile_plot_MTL"
if not os.path.exists(plot_output_dir):
    os.makedirs(plot_output_dir)

checkpoint_dir = "/home/hyb/MTL-FL/new/project_ppml/Project_SSS/compression-pipeline/MTL"

os.system("rm -r %s/*"%text_output_dir)

img_size_dict = {
    'cls': (1,224,224,3),
    'det': (1,320,416,3),
    'seg': (1,270,480,3),
    'pose': (1,256,256,3)
}

for ratio in [0.1]:
    # for task in ['cls', 'seg']:
    for task in ['cls','seg','det','pose']:
        # this_ckpt_path = os.path.join(checkpoint_dir, "new_%s_ckpts_tflite/ckpts_tflite"%task)
        tflite_path = os.path.join(checkpoint_dir, "mobilenetv2_%s_pretrain_%.1f.tflite"%(task,ratio))

        tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
        input_details = tflite_interpreter.get_input_details()
        tflite_interpreter.resize_tensor_input(input_details[0]['index'], img_size_dict[task])
        tflite_interpreter.allocate_tensors()
        input_details = tflite_interpreter.get_input_details()

        tflite_output_path = os.path.join(checkpoint_dir, "optimized_mobilenetv2_resized_%s_pretrain_%.1f.tflite"%(task,ratio))

        output_text_path = os.path.join(text_output_dir, "profile_%s_%.1f.txt"%(task, ratio))

        output_plot_path = os.path.join(plot_output_dir, "profile_%s_%.1f.png"%(task, ratio))

        cmd = "python tflite_tools.py -i %s -o %s --calc-macs --calc-size --plot %s >> %s"%(tflite_path, tflite_output_path, output_plot_path,output_text_path)

        os.system(cmd)

        os.system("echo '== Input details ==' >> %s"%output_text_path)
        os.system("echo 'shape %s' >> %s"%(input_details[0]['shape'], output_text_path))
    