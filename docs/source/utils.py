import pickle
import gzip

class utils:
  
  def save_model_compressed(self,model,filename):
      """
      :Description: 
      Save model weights & configurations into compressed format.
      :parameter model: object from class.
      :type model: object.
      parameter filename: file which saves into the model.
      :type filename: string.
      :returns: None.
      """ 

      filename_ = filename + ".gz"
      outfile = gzip.open(filename_,'wb')
      pickle.dump(model,outfile)
      outfile.close()

  def load_model_compressed(self,filename): 
      """
      :Description: 
      Load model weights & configurations into compressed format.
      parameter filename: file which loads into the model.
      :type filename: string.
      :returns: loaded_nn.
      """ 
      filename = filename+".gz"
      infile = gzip.open(filename,'rb')
      loaded_nn = pickle.load(infile)
      infile.close()
      return loaded_nn