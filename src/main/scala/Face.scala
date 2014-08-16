import org.opencv.core.Rect

/**
 * Created by chris on 11/08/14.
 */
class Face(
            val x:Int,
            val y:Int,
            val width: Int,
            val height: Int,
            val leftEye: Eye,
            val rightEye: Eye,
            val nose:Rect
) {



}
