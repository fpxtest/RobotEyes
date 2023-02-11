import subprocess


class Imagemagick(object):

    def __init__(self, img1, img2, diff):
        self.img1 = img1
        self.img2 = img2
        self.diff = diff

    def compare_images(self):
        compare_cmd = f'convert "{self.img1}" "{self.img2}" -metric RMSE -compare -format "%[distortion]" info:'
        attempts = 0
        while attempts < 2:
            proc = subprocess.Popen(compare_cmd,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

            out, err = proc.communicate()
            if err:
                print('Comparison err: %s' % err)
            if out:
                print('Comparison output: %s' % out)
            try:
                diff = out
                trimmed = float("{:.2f}".format(float(diff)))
                return trimmed
            except ValueError:
                if attempts == 0:
                    print('Comparison failed first time. Output %s' % err)
                    compare_cmd = 'magick ' + compare_cmd
                else:
                    raise Exception('Could not parse comparison output: %s' % err)
            finally:
                attempts += 1

    def compare_images_with_compare(self):
        compare_cmd = 'compare -metric RMSE -subimage-search -dissimilarity-threshold 1.0 "%s" "%s" "%s"' \
                      % (self.img1, self.img2, self.diff)

        attempts = 0
        while attempts < 2:
            proc = subprocess.Popen(compare_cmd,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

            out, err = proc.communicate()
            print('Comparison output: %s' % err)
            diff = err.split()[1][1:-1]

            try:
                trimmed = float("{:.2f}".format(float(diff)))
                return trimmed
            except ValueError:
                if attempts == 0:
                    print('Comparison failed first time. Output %s' % err)
                    compare_cmd = 'magick ' + compare_cmd
                else:
                    raise Exception('Could not parse comparison output: %s' % err)
            finally:
                attempts += 1

    def compare_images_to_output(self):
        compare_cmd = f'compare -fuzz 15% -metric ae {self.img1} {self.img2} {self.diff} '
        attempts = 0
        while attempts < 2:
            proc = subprocess.Popen(compare_cmd,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

            err, out = proc.communicate()
            if err:
                print('Comparison err: %s' % err)
            if out:
                print('Comparison output: %s' % out)
            try:
                diff = out
                trimmed = float("{:.2f}".format(float(diff)))
                return trimmed
            except Exception as e:
                if attempts == 0:
                    print('Comparison failed first time. Output %s' % err)
                else:
                    print(e)
                    raise Exception('Could not parse comparison output: %s' % err)
            finally:
                attempts += 1


if __name__ == '__main__':
    diff = Imagemagick('base.png', 'diff.png', 'out.png').compare_images_to_output()
    print(diff)