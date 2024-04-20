package com.example.final_app;

import androidx.appcompat.app.AppCompatActivity;
import com.example.final_app.databinding.ActivityCameraBinding;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.widget.Toast;

import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class ActivityCamera extends AppCompatActivity {
    public ActivityCameraBinding binding;
    public int id_current_camera;
    public String url_Current, title_camera;
    private boolean isImageVisible = false; // Définissez l'état initial de l'image à invisible

    private static final long FRAME_RATE = 60;
    private Handler handler;
    long frameDelay = 1000 / FRAME_RATE;
    String[] videoUrl = {"http://192.168.43.190/cam-lo.jpg", "", "", ""};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityCameraBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        binding.flecheTransitionGauche.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Logique pour passer à la vue de caméra précédente
                id_current_camera -= 1;
                if (id_current_camera < 0) {
                    id_current_camera = videoUrl.length - 1; // Revenir à la dernière caméra si l'indice est inférieur à 0
                }
                url_Current = videoUrl[id_current_camera / 4];
                Toast.makeText(ActivityCamera.this, "Flèche gauche cliquée", Toast.LENGTH_SHORT).show();
                title_camera = "Caméra numéro " + (id_current_camera + 1); // Ajoutez +1 pour obtenir le bon numéro de caméra
            }
        });

        binding.flecheTransitionDroite.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Logique pour passer à la vue de caméra suivante
                id_current_camera += 1;
                if (id_current_camera >= videoUrl.length) {
                    id_current_camera = 0; // Revenir à la première caméra si l'indice dépasse la dernière caméra disponible
                }
                url_Current = videoUrl[id_current_camera / 4];
                Toast.makeText(ActivityCamera.this, "Flèche droite cliquée", Toast.LENGTH_SHORT).show();
                title_camera = "Caméra numéro " + (id_current_camera + 1); // Ajoutez +1 pour obtenir le bon numéro de caméra
            }
        });

        // Configurez le bouton pour basculer la visibilité de l'image
        binding.toggleButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                toggleImageVisibility();
            }
        });

        handler = new Handler();
        // Assurez-vous que la valeur initiale de url_Current est valide
        url_Current = videoUrl[0];
        Update();
    }

    private void toggleImageVisibility() {
        if (isImageVisible) {
            // Si l'image est visible, la rendre invisible
            binding.videoView.setVisibility(View.GONE);
            title_camera = "Caméra numéro 1";
        } else {
            // Si l'image est invisible, chargez-la à partir des ressources locales et affichez-la
            binding.videoView.setImageResource(R.drawable.img);
            binding.videoView.setVisibility(View.VISIBLE);
            title_camera = "Caméra numéro 1";
        }
        // Inversez l'état de visibilité de l'image
        isImageVisible = !isImageVisible;
    }

    private void Update() {
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                new DownloadImageTask().execute(url_Current);
                // Mettez à jour le titre de la caméra ici si nécessaire
                binding.textTitleCamera.setText(title_camera);
                handler.postDelayed(this, frameDelay);
            }
        }, frameDelay);
    }

    private class DownloadImageTask extends AsyncTask<String, Void, Bitmap> {
        @Override
        protected Bitmap doInBackground(String... urls) {
            String imageUrl = urls[0];
            Bitmap bitmap = null;
            try {
                URL url = new URL(imageUrl);
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setDoInput(true);
                connection.connect();
                InputStream input = connection.getInputStream();
                bitmap = BitmapFactory.decodeStream(input);
            } catch (Exception e) {
                e.printStackTrace();
            }
            return bitmap;
        }

        @Override
        protected void onPostExecute(Bitmap result) {
            if (result != null) {
                binding.videoView.setImageBitmap(result);
            }
        }
    }
}
