# Import Packages
import os
import pickle

# DIRECTORY UTILS
def set_outdir(OUTPUT_DIR, args):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR+'./pth/'):
        os.makedirs(OUTPUT_DIR+'./pth/')
    if not os.path.exists(OUTPUT_DIR+'./'+args.dataset+'/'):
        os.makedirs(OUTPUT_DIR+'/'+args.dataset+'/')
    with open(OUTPUT_DIR + '/pth/'+args.dataset+'.pth', 'wb') as f:
        pickle.dump({' Model Arguments' : args}, f)