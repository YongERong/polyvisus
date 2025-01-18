import React, { useState, useEffect } from 'react';
import { StyleSheet, View, Text, Dimensions } from 'react-native';
import { Camera, useCameraDevices, useFrameProcessor } from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';

const { width, height } = Dimensions.get('window');

// Define interfaces for our state
interface Position {
    x: number;
    y: number;
}

interface KeyboardState {
    text: string;
    layout: string;
    shift_active: boolean;
    pressed_keys: string[];
    current_keys: { [key: string]: Position };
}

const App = () => {
    const [hasPermission, setHasPermission] = useState(false);
    const [keyboardState, setKeyboardState] = useState<KeyboardState>({
        text: "",
        layout: "letters",
        shift_active: false,
        pressed_keys: [],
        current_keys: {}
    });
    const devices = useCameraDevices();
    const device = devices.find((d) => d.position === 'back');

    const format = device?.formats.find(f =>
        f.maxFps >= 30 && f.videoHeight >= 720
    );

    useEffect(() => {
        checkPermission();
    }, []);

    const checkPermission = async () => {
        const cameraPermission = await Camera.requestCameraPermission();
        setHasPermission(cameraPermission === 'granted');
    };

    const fetchKeyboardState = async () => {
        try {
            const response = await fetch('http://172.20.10.11:5000/keyboard-state');
            const data = await response.json();
            setKeyboardState(data);
        } catch (error) {
            console.error('Error fetching keyboard state:', error);
        }
    };

    // Create a function that can be called from worklets to run your JS function
    const runUpdateKeyboardState = Worklets.createRunOnJS(fetchKeyboardState);

    // Then use it in your frame processor
    const frameProcessor = useFrameProcessor((frame) => {
        'worklet';
        // Call the runOnJS function from within the worklet
        runUpdateKeyboardState();
    }, []);

    const renderKey = (key: string, position: Position) => {
        const isPressed = keyboardState.pressed_keys.includes(key);
        const isSpecial = ['SHIFT', '123', 'ABC', 'SYM', 'SPACE', 'BACK', 'ENTER'].includes(key);

        return (
            <View
                key={key}
                style={[
                    styles.key,
                    {
                        left: (position.x / 600) * width,
                        top: (position.y / 400) * height,
                        backgroundColor: isPressed ? 'rgba(0, 255, 0, 0.3)' :
                            isSpecial ? 'rgba(0, 165, 255, 0.3)' :
                                'rgba(255, 255, 255, 0.3)',
                    }
                ]}
            >
                <Text style={styles.keyText}>
                    {keyboardState.shift_active && key.length === 1 ? key.toUpperCase() : key}
                </Text>
            </View>
        );
    };

    if (!hasPermission) {
        return <Text>No access to camera</Text>;
    }

    if (!device || !format) {
        return <Text>Loading...</Text>;
    }

    return (
        <View style={styles.container}>
            <Camera
                style={StyleSheet.absoluteFill}
                device={device}
                isActive={true}
                frameProcessor={frameProcessor}
                format={format}
                fps={30}
            >
                <View style={styles.overlay}>
                    {Object.entries(keyboardState.current_keys).map(([key, pos]) =>
                        renderKey(key, pos)
                    )}
                </View>
                <View style={styles.textContainer}>
                    <Text style={styles.typedText}>{keyboardState.text}</Text>
                </View>
            </Camera>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    overlay: {
        ...StyleSheet.absoluteFillObject,
    },
    key: {
        position: 'absolute',
        width: 40,
        height: 40,
        backgroundColor: 'rgba(255, 255, 255, 0.3)',
        borderRadius: 5,
        justifyContent: 'center',
        alignItems: 'center',
    },
    keyText: {
        color: 'white',
        fontSize: 16,
        fontWeight: 'bold',
    },
    textContainer: {
        position: 'absolute',
        bottom: 50,
        left: 20,
        right: 20,
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        padding: 10,
        borderRadius: 5,
    },
    typedText: {
        color: 'white',
        fontSize: 18,
    },
});

export default App;
