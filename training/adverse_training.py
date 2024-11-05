import tensorflow as tf
from sklearn.model_selection import train_test_split

saved_model_name = 'tf_model'
new_model = 'tf_model_adv_trained'

def create_adversarial_pattern(model, input_image, input_label, epsilon=0.1, alpha=0.01, num_iter=40):
    """
    Generates adversarial examples using PGD.

    Args:
    - model: The trained model.
    - input_image: The input image (as a Tensor).
    - input_label: The true label of the image.
    - epsilon: The maximum perturbation.
    - alpha: The step size.
    - num_iter: The number of iterations.

    Returns:
    - The adversarial example.
    """
    input_image = tf.convert_to_tensor(input_image)
    input_label = tf.convert_to_tensor(input_label)

    # Start with the original image as the initial adversarial image
    adv_image = tf.identity(input_image)

    # Perform PGD
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image)
            loss = tf.keras.losses.sparse_categorical_crossentropy(input_label, prediction)

        # Get the gradient of the loss w.r.t. the input image
        gradient = tape.gradient(loss, adv_image)
        signed_grad = tf.sign(gradient)

        # Update the adversarial image by taking a step in the direction of the gradient
        adv_image = adv_image + alpha * signed_grad

        # Clip the adversarial image to ensure it's within the epsilon-ball
        adv_image = tf.clip_by_value(adv_image, input_image - epsilon, input_image + epsilon)
        adv_image = tf.clip_by_value(adv_image, 0.0, 1.0)  # Ensure pixel values are in [0, 1]

    return adv_image


def adversarial_training(model, x_train, y_train, x_val, y_val, epochs, batch_size, epsilon, alpha, num_iter):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        for i in range(0, len(x_train), batch_size):
            # Get a batch of data
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Generate adversarial examples
            adv_x_batch = create_adversarial_pattern(model, x_batch, y_batch, epsilon=epsilon, alpha=alpha, num_iter=num_iter)

            # Combine original and adversarial examples
            x_combined_batch = tf.concat([x_batch, adv_x_batch], axis=0)
            y_combined_batch = tf.concat([y_batch, y_batch], axis=0)

            # Train on the combined dataset
            loss, acc = model.train_on_batch(x_combined_batch, y_combined_batch)

            if i % (batch_size * 10) == 0:  # Adjust the condition based on your dataset size
                print(f"Batch {i//batch_size}/{len(x_train)//batch_size}, Loss: {loss}, Accuracy: {acc}")


        # Evaluate on validation data
        val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=2)
        print(f"Validation accuracy after epoch {epoch+1}: {val_accuracy}")

# Hyperparameters
epsilon = 0.1  # Maximum perturbation
alpha = 0.01   # Step size
num_iter = 40  # Number of iterations
batch_size = 128
epochs = 5

model = tf.keras.models.load_model(saved_model_name)

mnist = tf.keras.datasets.mnist

(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

x_train_full = x_train_full / 255.0
x_train_full = x_train_full.reshape(x_train_full.shape[0], 28, 28, 1)
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# Assuming your data and model are already defined and compiled
adversarial_training(model, x_train, y_train, x_val, y_val, epochs, batch_size, epsilon, alpha, num_iter)

model.save(new_model)
