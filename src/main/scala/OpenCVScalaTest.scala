import java.awt.{Dimension, Color}
import java.io.{File, ByteArrayInputStream}
import java.util
import javax.imageio.ImageIO
import javax.swing._

import scala.collection.JavaConversions._

import org.opencv.highgui.{Highgui, VideoCapture}
import org.opencv.core._
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.{Objdetect, CascadeClassifier}
;

/**
 * Created by chris on 11/08/14.
 */
object OpenCVScalaTest {

  def WHITE = new Scalar(255, 255, 255);
  def BLACK = new Scalar(0, 0, 0);

  def getBiggestRect(rects:Array[Rect]): Rect = {
    var biggest: Rect = null;
    for( rect <- rects ) {
      if( biggest == null || biggest.area() < rect.area() ) {
        biggest = rect;
      }
    }
    return biggest;
  }

  def balanceCircle(cx: Int, cy: Int, r: Int, mat: Mat, tmp: Mat): Double = {
    val count = Math.PI * r * r;
    Core.rectangle(tmp, new Point(0, 0), new Point(tmp.width(), tmp.height()), BLACK, -1);
    Core.ellipse(tmp, new Point(cx, cy), new Size(r, r), 0, -90, 90, WHITE, -1);
    Core.bitwise_and(mat, tmp, tmp);
    val left = Core.sumElems(tmp);

    Core.rectangle(tmp, new Point(0, 0), new Point(tmp.width(), tmp.height()), BLACK, -1);
    Core.ellipse(tmp, new Point(cx, cy), new Size(r, r), 0, 90, -90, WHITE, -1);
    Core.bitwise_and(mat, tmp, tmp);
    val right = Core.sumElems(tmp);

    return 2 * (255 - Math.abs(left.`val`(0) - right.`val`(0)))/(255D * count);
  }

  def meanCircle(cx: Int, cy: Int, minRadius: Int, maxRadius: Int, mat: Mat, tmp: Mat, minProportion: Double, maxProportion: Double): Double = {
    val proportion = maxProportion - minProportion;
    Core.rectangle(tmp, new Point(0, 0), new Point(tmp.width(), tmp.height()), BLACK, -1);
    val c = new Point(cx, cy);
    Core.circle(tmp, c, maxRadius, WHITE, -1);
    var count = Math.PI * maxRadius * maxRadius;
    if( minRadius > 0 ) {
      Core.circle(tmp, c, minRadius, BLACK, -1);
      count -= Math.PI * minRadius * minRadius;
    }
    Core.rectangle(tmp, new Point(0, cy - maxRadius + maxRadius * 2 * (1 - minProportion)), new Point(tmp.width(), cy - maxRadius + maxRadius * 2 * (1 - maxProportion)), BLACK, -1);
    count *= proportion;
    Core.bitwise_and(mat, tmp, tmp);
    val total = Core.sumElems(tmp);
    return total.`val`(0) / count;
  }


  def analyseIris(fullGray:Mat, eyeArea:Rect, midMult:Double, edgeMult:Double, left:Boolean): Rect = {
    val lowerEyeArea = new Rect(eyeArea.x, eyeArea.y+(eyeArea.height*2)/8, eyeArea.width, (eyeArea.height*6)/8);
    val gray = fullGray.submat(lowerEyeArea)
    val inv = gray;
    Core.bitwise_not(gray, inv);
    var blockSize = Math.max(1, eyeArea.width / 5);
    if( blockSize % 2 == 0 ) {
      blockSize = blockSize + 1;
    }
    Imgproc.adaptiveThreshold(inv, inv, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, blockSize, 1);
    /*
    //Imgproc.blur(inv, inv, new Size(3, 3))
    Imgproc.Canny(inv, inv, 100, 255)

    val circles = new Mat();
    Imgproc.HoughCircles(
      inv,
      circles,
      Imgproc.CV_HOUGH_GRADIENT,
      10,
      inv.rows()/16
    );
    var col = 0;
    while( col < circles.cols ) {
      val circle = circles.get(0, col)
      Core.circle(inv, new Point(circle(0).toInt, circle(1).toInt), circle(2).toInt, WHITE)
      col += 1;
    }
    */

    val oval = Mat.zeros(gray.size(), CvType.CV_8UC1);
    var cDiv = 2.05;
    if( left ) {
      cDiv = 1.95;
    }
    val c = new Point(gray.width()/cDiv, gray.height()/2.6);
    var cAngle = 5;
    if( left ) {
      cAngle = -cAngle;
    }
    Core.ellipse(oval, c, new Size(gray.width()/4, gray.height()/3), cAngle, 0, 360, WHITE, -1);
    Core.bitwise_and(oval, inv, inv);

    var totalWeight = 0.0;
    var totalWeightedYOff = 0.0;
    val columnWeights = new Array[Double](inv.width());

    // 0 horizontal 1 = diagonal (along eye box)
    var eyeTilt = -0.2;
    if( left ) {
      eyeTilt = -eyeTilt;
    }
    val blinkMidY = gray.height()/2.7;

    var blinkCyStart = 0.0;
    var blinkCyEnd = 0.0;

    var x = inv.width()
    while( x>0 ) {
      x = x-1;
      val dx = x - c.x;
      val mult = (Math.abs(dx) * (edgeMult - midMult) * 2)/inv.width() + midMult;

      var columnWeight = 0.0;
      val blinkCy = (dx / c.x) * eyeTilt * lowerEyeArea.height/2 + blinkMidY;
      if( x == 0 ) {
        blinkCyStart = blinkCy;
      } else if( x == inv.width() - 1 ) {
        blinkCyEnd = blinkCy;
      }
      var y = inv.height();
      while( y > 0 ) {
        y = y - 1;
        // TODO reuse this array
        val values = inv.get(y, x);
        val p = values(0);
        if( p > 0 ) {
          totalWeight += p;
          val dy = y - blinkCy;
          totalWeightedYOff += (dy * p);
          columnWeight += p;
        }
      }
      columnWeights(x) = columnWeight * mult;
    }

    val blinkY = (totalWeightedYOff * 2) / totalWeight;

    // debug only
    //inv.copyTo(gray);
    val big = new Mat();
    Imgproc.resize(inv, big, new Size(inv.width() * 2, inv.height() * 2));
    Imgproc.cvtColor(big, big, Imgproc.COLOR_GRAY2RGBA);
    Core.line(big, new Point(0, blinkCyStart*2), new Point(big.width(), blinkCyEnd*2), new Scalar(255, 0, 0), 1);

    var result: Rect = null;

    if( blinkY <= 0 ) {

      val d = Math.round((gray.height()*2)/3.3f);

      var x = gray.width()/2;
      var y = gray.height()/2;
      var maxMean = 0.0;
      val r = d/2;
      val highlightR = 0;
      val minPupilR = r/4;
      val whiteR = (r * 3) / 2;
      val tmp = new Mat(inv.rows(), inv.cols(), inv.`type`());

      var cx = r;
      while( cx<gray.width()-r ) {
        cx = cx + 1
        val dx = x - c.x;
        val mult = (Math.abs(dx) * (edgeMult - midMult) * 2)/inv.width() + midMult;
        var cy = r;
        while( cy<gray.height()-r ) {
          cy = cy + 1;

          //                    ArrayList<Double> values = extractValues(cx, cy, highlightR, r, inv, true);
          //                    double mean = mean(values, null);
          val irisMean = meanCircle(cx, cy, highlightR, r, inv, tmp, 0.65, 1);
          val whiteMean = 255 - meanCircle(cx, cy, r, whiteR, inv, tmp, 0.5, 1);
          val pupilMean = 255 - meanCircle(cx, cy, 0, minPupilR, gray, tmp, 0.5, 0.75);
          var mean = (Math.PI * irisMean * r * r) + (Math.PI * pupilMean * minPupilR * minPupilR * mult) + (Math.PI * whiteMean * (whiteR * whiteR - r * r));
          if( mean > maxMean ) {
            val balance = balanceCircle(cx, cy, whiteR, inv, tmp);
            mean *= Math.sqrt(balance);
            // check the balance
            if( mean > maxMean ) {
              x = cx;
              y = cy;
              maxMean = mean;
            }
          }
        }

      }
      result = new Rect(
        lowerEyeArea.x + x - r,
        lowerEyeArea.y + y - r,
        d,
        d
      );

    }

    return result;
  }

  def main(args: Array[String]) = {
    System.load(new File("lib/libopencv_java249.dylib").getAbsolutePath())
    println("creating frame...")
    val frame = new JFrame("Video Capture Test")
    frame.setDefaultCloseOperation(WindowConstants.HIDE_ON_CLOSE)
    val view = new JPanel()
    frame.setContentPane(view)

    println("starting...")
    val videoCapture = new VideoCapture(0)

    println("...started")
    var first = true;

    val faceDetector = new CascadeClassifier(getClass.getResource("haarcascade_frontalface_default.xml").getPath)
    val leftEyeDetector = new CascadeClassifier(getClass.getResource("haarcascade_mcs_lefteye.xml").getPath)
    val rightEyeDetector = new CascadeClassifier(getClass.getResource("haarcascade_mcs_righteye.xml").getPath)

    val thread = new Thread(new Runnable {
      override def run(): Unit = {
        val temp = new Mat()
        val bytes = new MatOfByte()
        val ext = ".png"
        while( videoCapture.grab && (frame.isVisible || first) ) {
          while( !videoCapture.read(temp) ) {
            println("...waiting...")
          }
          val mat = new Mat();
          Imgproc.cvtColor(temp, mat, Imgproc.COLOR_RGB2GRAY)
          val width = mat.cols()
          val height = mat.rows()

          // do some analysis
          val faceRects = new MatOfRect()
          // TODO base sizes off the preview size
          faceDetector.detectMultiScale(
            mat,
            faceRects,
            1.10,
            5,
            Objdetect.CASCADE_SCALE_IMAGE,
            new Size(100, 100),
            new Size(400, 400)
          )


          val faceRectsArray = faceRects.toArray;
          var faceCount = 0;
          val faces = new Array[Face](faceRectsArray.length)
          for( faceRect <- faceRectsArray ) {
            val leftEyeSampleRect = new Rect(
              faceRect.x,
              (faceRect.y + faceRect.height/4.5).toInt,
              faceRect.width/2,
              faceRect.height/3
            );
            val leftEyeFaceMat = new Mat(mat, leftEyeSampleRect)
            val leftEyeRects = new MatOfRect();
            leftEyeDetector.detectMultiScale(
              leftEyeFaceMat,
              leftEyeRects,
              1.15,
              2,
              Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE,
              new Size(20, 20),
              new Size()
            );
            val leftEyeRectsArray = leftEyeRects.toArray
            var leftEye: Eye = null;
            if( leftEyeRectsArray.length > 0 ) {
              val leftEyeRect = getBiggestRect(leftEyeRectsArray);

              val leftEyeMat = new Mat(leftEyeFaceMat, leftEyeRect);

              //Imgproc.cvtColor(new Mat(leftEyeFaceMat, leftEyeRect), leftEyeMat, Imgproc.COLOR_RGB2GRAY);

              val circles = new Mat();

              //Core.bitwise_not(leftEyeMat, leftEyeMat)
              //Imgproc.adaptiveThreshold(leftEyeMat, leftEyeMat, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 11, 2)
              //Imgproc.blur(leftEyeMat, leftEyeMat, new Size(3, 3))
              //Imgproc.Canny(leftEyeMat, leftEyeMat, 100, 255)

//              val contours = new util.ArrayList[MatOfPoint]();
//              Imgproc.findContours(leftEyeMat, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
//              Imgproc.drawContours(leftEyeMat, contours, -1, new Scalar(255, 255, 255))
//              Imgproc.HoughCircles(
//                leftEyeMat,
//                circles,
//                Imgproc.CV_HOUGH_GRADIENT,
//                10,
//                leftEyeMat.rows()/16
//              );
              //              Imgproc.HoughCircles(
              //                leftEyeMat,
              //                circles,
              //                Imgproc.CV_HOUGH_GRADIENT,
              //                1,
              //                leftEyeMat.rows()/8.0);

              var pupilCX = leftEyeRect.width/2;
              var pupilCY = leftEyeRect.height/2;
              var pupilR = 0;
              val pupil = analyseIris(leftEyeFaceMat, leftEyeRect, 1, 0.6, true);
              if( pupil != null ) {
                pupilCX = leftEyeSampleRect.x + pupil.x + pupil.width/2;
                pupilCY = leftEyeSampleRect.y + pupil.y + pupil.height/2;
                pupilR = pupil.height / 2;
              }

              leftEye = new Eye(
                leftEyeSampleRect.x + leftEyeRect.x,
                leftEyeSampleRect.y + leftEyeRect.y,
                leftEyeRect.width,
                leftEyeRect.height,
                pupilCX,
                pupilCY,
                pupilR
              );
            }
            val rightEyeSampleRect = new Rect(
              faceRect.x + faceRect.width/2,
              (faceRect.y + faceRect.height/4.5).toInt,
              faceRect.width/2,
              faceRect.height/3
            );
            val rightEyeFaceMat = new Mat(mat, rightEyeSampleRect)
            val rightEyeRects = new MatOfRect();
            rightEyeDetector.detectMultiScale(
              rightEyeFaceMat,
              rightEyeRects,
              1.15,
              2,
              Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE,
              new Size(20, 20),
              new Size()
            );
            val rightEyeRectsArray = rightEyeRects.toArray
            var rightEye: Eye = null;
            if( rightEyeRectsArray.length > 0 ) {
              val rightEyeRect = getBiggestRect(rightEyeRectsArray);
              rightEye = new Eye(
                rightEyeSampleRect.x + rightEyeRect.x,
                rightEyeSampleRect.y + rightEyeRect.y,
                rightEyeRect.width,
                rightEyeRect.height,
                0,
                0,
                0
              );
            }

            val face = new Face(faceRect.x, faceRect.y, faceRect.width, faceRect.height, leftEye, rightEye)
            faces(faceCount) = face;
            faceCount = faceCount + 1;
          }

          Highgui.imencode(ext, mat, bytes)
          val ins = new ByteArrayInputStream(bytes.toArray)
          val image = ImageIO.read(ins)

          SwingUtilities.invokeLater(new Runnable {
            override def run(): Unit = {
              if( first ) {
                println("size ("+mat.cols()+","+mat.rows()+")")
                val dimension = new Dimension(width, height)
                view.setSize(dimension)
                view.setMinimumSize(dimension)
                view.setMaximumSize(dimension)
                view.setPreferredSize(dimension)
                frame.pack()
                frame.setVisible(true)

                first = false
              }
              val graphics = view.getGraphics
              graphics.drawImage(image, 0, 0, null)

              for( face <- faces ) {
                graphics.setColor(Color.RED);
                graphics.drawRect(face.x, face.y, face.width, face.height)
                if( face.leftEye != null ) {
                  graphics.setColor(Color.GREEN);
                  graphics.drawRect(face.leftEye.x, face.leftEye.y, face.leftEye.width, face.leftEye.height)
                  graphics.setColor(Color.RED);
                  graphics.drawOval(
                    face.leftEye.pupilCX - face.leftEye.pupilR,
                    face.leftEye.pupilCY - face.leftEye.pupilR,
                    face.leftEye.pupilR * 2,
                    face.leftEye.pupilR * 2
                  );
                }
                if( face.rightEye != null ) {
                  graphics.setColor(Color.BLUE);
                  graphics.drawRect(face.rightEye.x, face.rightEye.y, face.rightEye.width, face.rightEye.height)
                }

              }
              graphics.dispose()
            }
          })
        }
        println("DONE!")
        System.exit(0)
      }
    })
    thread.start()

  }

}
