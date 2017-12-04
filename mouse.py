import win32api
import win32con
import Leap,time,thread,sys

class SampleLisener(Leap.Listener):
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']

    def on_init(self, controller):
        print "Initialized"

    def on_connect(self, controller):
        print "Connected"

    def on_disconnect(self, controller):
        print "Disconnected"

    def on_exit(self, controller):
        print "Exited"

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()

        # Get hands
        for hand in frame.hands:
            mouse_movement_status = True
            controller.enable_gesture(Leap.Gesture.TYPE_CIRCLE)
            controller.config.set("Gesture.Circle.MinRadius", 10.0)
            controller.config.set("Gesture.Circle.MinArc", .5)
            controller.config.save()

            if len(frame.fingers.extended()) == 5:
                for gesture in frame.gestures():
                    if gesture.type == Leap.Gesture.TYPE_CIRCLE:
                        mouse_movement_status = False
                        circle = Leap.CircleGesture(gesture)
                        if circle.pointable.direction.angle_to(circle.normal) <= Leap.PI / 2:
                            win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, 10, 0)
                        else:
                            win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, -10, 0)
                    else:
                        mouse_movement_status = True

            self.mouse_move(frame.hands[0], mouse_movement_status)


    def convert_velocity_to_mouse_movement(self, dx, dy):
        k = 0.04
        x_movement = int(dx * k)
        y_movement = int(dy * k)
        return x_movement, y_movement

    def mouse_move(self, hand, status):
        if status:
            dx = float(hand.palm_velocity[0])
            dy = float(hand.palm_velocity[2])
            mouse_movement = self.convert_velocity_to_mouse_movement(dx, dy)
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, mouse_movement[0], mouse_movement[1], 0, 0)


def main():
    # Create a sample listener and controller
    listener = SampleLisener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)



    # Keep this process running until Enter is pressed
    print "Press Enter to quit..."
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)

if __name__ == "__main__":
    main()